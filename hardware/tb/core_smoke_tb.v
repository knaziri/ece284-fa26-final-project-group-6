// End-to-end self-checking smoke test for core.v (all 7 stages of the project).
// Random 4-bit weights (signed) and activations (unsigned); golden computed here,
// no input files needed. Sizes mirror features[27]: nij=36 activations per kij,
// 9 kij, 8x8 array.
//
//   for each kij:  reset -> weights XMEM->L0 -> kernel load -> acts XMEM->L0
//                  -> execute -> OFIFO->PMEM            (check every psum)
//   for each nij:  SFU acc of 9 psums -> store back      (check raw sum)
//   for each nij:  SFU load 1 word -> ReLU -> store back (check ReLU)
//
// Run:  iverilog -g2012 -o smoke.vvp -c filelist tb/core_smoke_tb.v && vvp smoke.vvp [+vcd]
`timescale 1ns/1ps

module core_smoke_tb;

  parameter bw = 4;
  parameter psum_bw = 16;
  parameter col = 8;
  parameter row = 8;
  parameter N = 36;     // activation vectors (nij) per kij
  parameter K = 9;      // kij

  localparam XMEM_W_BASE   = 1024;  // weights: 1024 + k*col + c ; acts: k*N + n
  localparam PMEM_ACC_BASE = 512;   // accumulated results
  localparam PMEM_RELU_BASE= 768;   // ReLU results

  reg clk = 0;
  reg reset = 1;
  always #0.5 clk = ~clk;

  // ---- instruction fields (driven on negedge) ----
  reg relu = 0, sfu_wb = 0, acc = 0;
  reg CEN_pmem = 1, WEN_pmem = 1;  reg [10:0] A_pmem = 0;
  reg CEN_xmem = 1, WEN_xmem = 1;  reg [10:0] A_xmem = 0;
  reg ofifo_rd = 0, l0_rd = 0, l0_wr = 0, execute = 0, load = 0;
  reg [row*bw-1:0] D_xmem = 0;

  wire [35:0] inst = {relu, sfu_wb, acc, CEN_pmem, WEN_pmem, A_pmem,
                      CEN_xmem, WEN_xmem, A_xmem, ofifo_rd, 1'b0, 1'b0,
                      l0_rd, l0_wr, execute, load};

  wire ofifo_valid;
  wire [psum_bw*col-1:0] sfp_out;

  core #(.bw(bw), .psum_bw(psum_bw), .col(col), .row(row)) dut (
      .clk(clk), .reset(reset), .inst(inst), .ofifo_valid(ofifo_valid),
      .D_xmem(D_xmem), .sfp_out(sfp_out));

  // ---- test data and golden ----
  integer W   [0:K-1][0:col-1][0:row-1];   // signed -8..7
  integer A   [0:K-1][0:N-1][0:row-1];     // unsigned 0..15
  integer P   [0:K-1][0:N-1][0:col-1];     // per-kij psum
  integer S   [0:N-1][0:col-1];            // sum over kij

  integer k, n, c, r, t, errors, neg_cnt, seed;

  function signed [psum_bw-1:0] lane(input [psum_bw*col-1:0] v, input integer i);
    lane = v[psum_bw*i +: psum_bw];
  endfunction

  task idle(input integer cycles);
    integer j;
    for (j=0; j<cycles; j=j+1) @(negedge clk);
  endtask

  task clear_ctrl;
    begin
      relu = 0; sfu_wb = 0; acc = 0;
      CEN_pmem = 1; WEN_pmem = 1; CEN_xmem = 1; WEN_xmem = 1;
      ofifo_rd = 0; l0_rd = 0; l0_wr = 0; execute = 0; load = 0;
    end
  endtask

  // XMEM -> L0 for `cnt` consecutive words starting at `base`
  // (XMEM read data is valid one cycle after the address, so l0_wr lags by 1)
  task xmem_to_l0(input integer base, input integer cnt);
    integer j;
    begin
      for (j=0; j<=cnt; j=j+1) begin
        @(negedge clk);
        if (j < cnt) begin CEN_xmem = 0; WEN_xmem = 1; A_xmem = base + j; end
        else         begin CEN_xmem = 1; end
        l0_wr = (j > 0);
      end
      @(negedge clk); clear_ctrl;
    end
  endtask

  initial begin
    if ($test$plusargs("vcd")) begin
      $dumpfile("core_smoke_tb.vcd");
      $dumpvars(0, core_smoke_tb);
    end

    seed = 284;
    errors = 0;
    neg_cnt = 0;

    for (k=0; k<K; k=k+1) begin
      for (c=0; c<col; c=c+1)
        for (r=0; r<row; r=r+1)
          W[k][c][r] = ($random(seed) & 15) - 8;
      for (n=0; n<N; n=n+1)
        for (r=0; r<row; r=r+1)
          A[k][n][r] = $random(seed) & 15;
    end
    for (n=0; n<N; n=n+1)
      for (c=0; c<col; c=c+1) begin
        S[n][c] = 0;
        for (k=0; k<K; k=k+1) begin
          P[k][n][c] = 0;
          for (r=0; r<row; r=r+1) P[k][n][c] = P[k][n][c] + A[k][n][r]*W[k][c][r];
          S[n][c] = S[n][c] + P[k][n][c];
        end
        if (S[n][c] < 0) neg_cnt = neg_cnt + 1;
      end

    // ================= Stage 1: DRAM -> XMEM (weights + activations) ==========
    idle(2);
    @(negedge clk); reset = 0;
    for (k=0; k<K; k=k+1) begin
      for (n=0; n<N; n=n+1) begin
        @(negedge clk);
        CEN_xmem = 0; WEN_xmem = 0; A_xmem = k*N + n;
        for (r=0; r<row; r=r+1) D_xmem[bw*r +: bw] = A[k][n][r];
      end
      for (c=0; c<col; c=c+1) begin
        @(negedge clk);
        CEN_xmem = 0; WEN_xmem = 0; A_xmem = XMEM_W_BASE + k*col + c;
        for (r=0; r<row; r=r+1) D_xmem[bw*r +: bw] = W[k][c][r];
      end
    end
    @(negedge clk); clear_ctrl;

    for (k=0; k<K; k=k+1) begin
      // fresh PE weights / FIFOs for this kij
      @(negedge clk); reset = 1;
      @(negedge clk); reset = 0;

      // ================= Stage 2: kernel XMEM -> L0 -> PE registers ===========
      xmem_to_l0(XMEM_W_BASE + k*col, col);
      for (t=0; t<col; t=t+1) begin
        @(negedge clk); l0_rd = 1; load = 1;
      end
      @(negedge clk); clear_ctrl;
      idle(row + col);

      // ================= Stage 3: activations XMEM -> L0 ======================
      xmem_to_l0(k*N, N);

      // ================= Stage 4: execution =================================
      for (t=0; t<N; t=t+1) begin
        @(negedge clk); l0_rd = 1; execute = 1;
      end
      @(negedge clk); clear_ctrl;
      idle(row + col + 4);

      if (!ofifo_valid) begin
        $display("ERROR kij=%0d: OFIFO not valid after execution", k);
        errors = errors + 1;
      end

      // ================= Stage 5: OFIFO -> PMEM ==============================
      // ofifo rd is registered inside ofifo.v, so entry j is on the output in
      // the cycle after rd: write PMEM one cycle behind rd.
      for (t=0; t<=N; t=t+1) begin
        @(negedge clk);
        ofifo_rd = (t < N);
        sfu_wb = 0;
        if (t > 0) begin CEN_pmem = 0; WEN_pmem = 0; A_pmem = k*N + t - 1; end
        else       begin CEN_pmem = 1; WEN_pmem = 1; end
      end
      @(negedge clk); clear_ctrl;

      for (n=0; n<N; n=n+1)
        for (c=0; c<col; c=c+1)
          if (lane(dut.pmem_instance.memory[k*N+n], c) !== P[k][n][c]) begin
            if (errors < 20)
              $display("PSUM mismatch kij=%0d nij=%0d col=%0d: hw=%0d exp=%0d",
                       k, n, c, lane(dut.pmem_instance.memory[k*N+n], c), P[k][n][c]);
            errors = errors + 1;
          end
    end
    $display("Stages 1-5 done: %0d psum words checked, errors so far = %0d", K*N, errors);

    // ================= Stage 6: accumulation in SFU -> store back =============
    for (n=0; n<N; n=n+1) begin
      for (t=0; t<=K; t=t+1) begin
        @(negedge clk);
        if (t < K) begin CEN_pmem = 0; WEN_pmem = 1; A_pmem = t*N + n; end
        else       begin CEN_pmem = 1; end
        acc = (t > 0);            // PMEM read data arrives one cycle after the address
      end
      @(negedge clk);             // store the raw sum (no ReLU)
      acc = 0; relu = 0; sfu_wb = 1;
      CEN_pmem = 0; WEN_pmem = 0; A_pmem = PMEM_ACC_BASE + n;
      @(negedge clk); clear_ctrl;
    end

    // ================= Stage 7: ReLU in SFU -> store back =====================
    for (n=0; n<N; n=n+1) begin
      @(negedge clk); CEN_pmem = 0; WEN_pmem = 1; A_pmem = PMEM_ACC_BASE + n;
      @(negedge clk); CEN_pmem = 1; acc = 1;     // single-cycle burst = load
      @(negedge clk);
      acc = 0; relu = 1; sfu_wb = 1;
      CEN_pmem = 0; WEN_pmem = 0; A_pmem = PMEM_RELU_BASE + n;
      @(negedge clk); clear_ctrl;
    end
    idle(2);

    for (n=0; n<N; n=n+1)
      for (c=0; c<col; c=c+1) begin
        if (lane(dut.pmem_instance.memory[PMEM_ACC_BASE+n], c) !== S[n][c]) begin
          if (errors < 40)
            $display("ACC mismatch nij=%0d col=%0d: hw=%0d exp=%0d",
                     n, c, lane(dut.pmem_instance.memory[PMEM_ACC_BASE+n], c), S[n][c]);
          errors = errors + 1;
        end
        if (lane(dut.pmem_instance.memory[PMEM_RELU_BASE+n], c) !== (S[n][c] < 0 ? 0 : S[n][c])) begin
          if (errors < 40)
            $display("RELU mismatch nij=%0d col=%0d: hw=%0d exp=%0d",
                     n, c, lane(dut.pmem_instance.memory[PMEM_RELU_BASE+n], c), S[n][c]);
          errors = errors + 1;
        end
      end

    $display("Checked %0d psums, %0d accumulated and %0d ReLU outputs (%0d negative sums).",
             K*N*col, N*col, N*col, neg_cnt);
    if (errors == 0) $display("PASS: 0 mismatches");
    else             $display("FAIL: %0d mismatches", errors);
    $finish;
  end

endmodule
