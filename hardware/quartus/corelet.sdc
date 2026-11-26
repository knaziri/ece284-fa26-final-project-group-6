# Timing constraints for corelet (Part 1, Step 5).
# CLK_PERIOD is the target (125 MHz, just under the achieved slow-corner Fmax); power is reported at this clock.
# The achieved Fmax is read from the TimeQuest "Slow 1200mV 100C Model" report.
set CLK_PERIOD 8.000
create_clock -name clk -period $CLK_PERIOD [get_ports clk]
derive_clock_uncertainty
