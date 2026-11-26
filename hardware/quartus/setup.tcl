# Creates the Quartus project for corelet (Part 1, Step 5).
# Usage (from hardware/quartus):
#   quartus_sh -t setup.tcl
#   quartus_sh --flow compile corelet
package require ::quartus::project

project_new corelet -overwrite

set_global_assignment -name FAMILY "Cyclone IV GX"
set_global_assignment -name DEVICE EP4CGX150DF31I7AD
set_global_assignment -name TOP_LEVEL_ENTITY corelet
set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files
set_global_assignment -name NUM_PARALLEL_PROCESSORS ALL

# Only corelet and what it instantiates (the SRAMs in core.v stay off the FPGA)
foreach f {corelet sfu mac_array mac_row mac_tile mac l0 ofifo fifo_depth64
           fifo_mux_2_1 fifo_mux_8_1 fifo_mux_16_1} {
    set_global_assignment -name VERILOG_FILE ../verilog/$f.v
}
set_global_assignment -name SDC_FILE corelet.sdc

# Power: vectorless estimation with 20% toggle rate on the input pins
set_global_assignment -name FLOW_ENABLE_POWER_ANALYZER ON
set_global_assignment -name POWER_DEFAULT_INPUT_IO_TOGGLE_RATE "20 %"
set_global_assignment -name POWER_USE_PVA ON

project_close
