# VCK190 (production silicon) Bare Platform (2021.2)
This bare platform is designed to provide basic linux (petalinux) embedded environment to test mlir generated physical design on the vck190 board (production silicon). It is built with 2021.2 vitis/ vivado/ petalinux tools and consists of a relatively empty Versal deign (CIPS, AI Engines, clock/ reset, NoC, BRAM). The build process consist of 3 steps: vivado, petalinux, aie_platform.

1. Vivado design (after 2020.1, we can no longer make explicit connections between PL and AIE in Vivado; instead we create a platform for Vitis to elaborate the adf graph design). This step generates the .xsa file.
2. Petalinux design which takes the .xsa as input. It creates an embedded platform package + sysroot.
3. Vitis design which compiles a 32 GMIO AIE design with simple add functionality to enable all NoC NMU/NSU connections so all shimDMAs (used for GMIO) are enabled. This rebuilds the design targeting the platform generated by steps 1-2. 

## Build steps
To build the sd card image and sysroot of this design, simply call:
```
make all
```

## Tool requirements
To run these build scripts, you should only need Vitis/Vivado/Petalinux 2021.2 (note the sw requirements of each tool as well). There was an issue where the program 'dot' (which is used by aiecompiler) needed to be installed. You can do that with:
`sudo apt-get install graphviz`

Once you've run the top level make, you should have sd_card/sd_card.img along with a sysroot which should be found at:
`aie_platform/sw_comp/sysroots/cortexa72-cortexa53-xilinx-linux`

## Build custom mlir designs
You can then follow the commands like those in the unit tests by calling the python wrapped build script like:
```
aiecc.py -v --aie-generate-xaiev2 --sysroot=<sysroot from the build> ./aie.mlir -I<mlir-aie install area>/runtime_lib <mlir-aie install area>/runtime_lib/test_library.cpp ./test.cpp -o test.elf
```
Note that `--aie-generate-xaiev2` is needed in order to generate the v2 drivers which vitis 2021.2 require.

## Copy files to the board
You can then copy the generated host and AIE elf files from your application to the board via an ethernet connection (check ip addr of board):
```
scp *elf root@192.168.0.101:/home/root/.  <-- check for the ip address of your board
```
Finally, you can then run the execuable on the board to verify functionality and performance!
```
ssh root@192.168.0.101
./test.elf
```