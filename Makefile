# ---------------------------------[ App ]------------------------------------ #
# Default application name is the same as the folder it is in.
# This can be overridden here if something different is required
APPNAME ?= $(notdir $(shell pwd))

# ---------------------------[ Build overrides ]------------------------------ #
MV_SOC_PLATFORM ?= myriad2
MV_SOC_REV      ?= ma2450

# Verbose or not
ECHO ?= @

LEON_RT_BUILD = no
DUAL_CPU_MODE = no
PROFILE_MODE = no

ifeq (yes, $(DUAL_CPU_MODE))
LEON_RT_BUILD = yes
endif
ifeq (yes, $(PROFILE_MODE))
LEON_RT_BUILD = yes
endif
lrtApp = $(DirAppObjBase)leonrt/

MV_SOC_OS = rtems
RTEMS_BUILD_NAME = b-prebuilt

srvPort ?= 30001 
MV_COMMON_BASE  ?= /home/fivosts/mdk_release_17.11.9_general_purpose/mdk/common

# ----------------------------[ Components used ]----------------------------- #
ComponentList_LOS := MV0212 VcsHooks Fp16Convert PipePrint
ifeq (yes, $(LEON_RT_BUILD))
lrt_build:
ComponentList_LRT := MV0212 PipePrint
endif
SHAVE_COMPONENTS = yes
ComponentList_SVE += kernelLib/MvCV

# --------------------[ Local shave applications sources ]-------------------- #
MvTensorApp = $(DirAppObjBase)shave/cmx/CNN
SHAVE_C_SOURCES_mvTensor = $(wildcard $(DirAppRoot)/shave/cmx/*.c)

SHAVE_GENASMS_mvTensor = $(patsubst %.c,                        \
                                    $(DirAppObjBase)%.asmgen,   \
                                    $(SHAVE_C_SOURCES_mvTensor) \
                          )

SHAVE_mvTensor_OBJS = $(patsubst $(DirAppObjBase)%.asmgen,  \
                                 $(DirAppObjBase)%_shave.o, \
                                 $(SHAVE_GENASMS_mvTensor)  \
                       )

PROJECTCLEAN += $(SHAVE_GENASMS_mvTensor)   \
                $(SHAVE_mvTensor_OBJS)

PROJECTINTERM += $(SHAVE_GENASMS_mvTensor)

ddrApp = $(DirAppObjBase)shave/ddr/doubleFunction
SHAVE_C_SOURCES_ddrApp = $(wildcard $(DirAppRoot)/shave/ddr/*.c)

SHAVE_ASM_SOURCES_ddrApp = $(wildcard $(DirAppRoot)/shave/ddr/asm/*.asm)

SHAVE_GENASMS_ddrApp = $(patsubst %.c,                          \
                                  $(DirAppObjBase)%.asmgen,     \
                                  $(SHAVE_C_SOURCES_ddrApp)     \
                        )

SHAVE_ddrApp_OBJS = $(patsubst $(DirAppObjBase)%.asmgen,    \
                               $(DirAppObjBase)%_shave.o,   \
                               $(SHAVE_GENASMS_ddrApp)      \
                     )

SHAVE_ddrApp_OBJS += $(patsubst %.asm,                       \
                                $(DirAppObjBase)%_shave.o,   \
                                $(SHAVE_ASM_SOURCES_ddrApp)  \
                      )

PROJECTCLEAN += $(SHAVE_GENASMS_ddrApp)     \
                $(SHAVE_ddrApp_OBJS)        \
                $(ddrApp).mvlib             \
                $(ddrApp).shvdlib           \
                $(ddrApp)_sym.o             \
                $(ddrApp)_bin.o             \
                $(ddrApp)_shvdlib.text      \
                $(ddrApp)_shvdlib.data      \
                $(ddrApp)_shvdlib.combined	\
                $(lrtApp)leonRTApp.mvlib 	\
                $(lrtApp)leonRTApp.rtlib 	

PROJECTINTERM += $(SHAVE_GENASMS_ddrApp)

RAWDATAOBJECTFILES += $(ddrApp)_sym.o $(ddrApp)_bin.o

# -----------------------[ Shave applications section ]----------------------- #
SHAVE_APP_LIBS = $(MvTensorApp).mvlib
SHAVE0_APPS    = $(MvTensorApp).shv0lib
SHAVE1_APPS    = $(MvTensorApp).shv1lib
SHAVE2_APPS    = $(MvTensorApp).shv2lib
SHAVE3_APPS    = $(MvTensorApp).shv3lib
SHAVE4_APPS    = $(MvTensorApp).shv4lib
SHAVE5_APPS    = $(MvTensorApp).shv5lib
SHAVE6_APPS    = $(MvTensorApp).shv6lib
SHAVE7_APPS    = $(MvTensorApp).shv7lib
SHAVE8_APPS    = $(MvTensorApp).shv8lib
SHAVE9_APPS    = $(MvTensorApp).shv9lib
SHAVE10_APPS   = $(MvTensorApp).shv10lib
SHAVE11_APPS   = $(MvTensorApp).shv11lib

# ----------------------------[ Tools overrides ]----------------------------- #

# Specific Linker
LinkerScript    =  ./scripts/ld/custom.ldscript

# Include the generic Makefile
include $(MV_COMMON_BASE)/generic.mk

# ------------------[ Local shave applications build rules ]------------------ #


ENTRYPOINTS1 = -e shave_conv -u shave_pool -u shave_fc -u shave_lrn -u shave_im2col --gc-sections
$(MvTensorApp).mvlib : $(SHAVE_mvTensor_OBJS) $(PROJECT_SHAVE_LIBS)
	$(ECHO) $(LD) $(MVLIBOPT) $(ENTRYPOINTS1) \
                              $(SHAVE_mvTensor_OBJS)    \
                              $(PROJECT_SHAVE_LIBS)     \
                              $(CompilerANSILibs)       -o $@

ENTRYPOINTS2 = 
$(ddrApp).mvlib : $(SHAVE_ddrApp_OBJS) $(PROJECT_SHAVE_LIBS)
	$(ECHO) $(LD) $(MVLIBOPT) $(ENTRYPOINTS2) \
                              $(SHAVE_ddrApp_OBJS)      \
                              $(PROJECT_SHAVE_LIBS)     \
                              $(CompilerANSILibs)       -o $@

$(ddrApp)_shvdlib.text : $(ddrApp).shvdlib
	$(ECHO) $(OBJCOPY) -O binary --only-section=.dyn.text $< $@

$(ddrApp)_shvdlib.data : $(ddrApp).shvdlib
	$(ECHO) $(OBJCOPY) -O binary --only-section=.dyn.data $< $@

$(ddrApp)_shvdlib.combined : $(ddrApp)_shvdlib.text $(ddrApp)_shvdlib.data
	$(ECHO) (difference=$(shell cat $(ddrApp)_shvdlib.text | wc -c);       \
             dd if=/dev/zero bs=1 count=$$((0x00100000 - $$difference))    \
             >> $(ddrApp)_shvdlib.text)
	$(ECHO) cat $(ddrApp)_shvdlib.text $(ddrApp)_shvdlib.data > $@


$(ddrApp)_bin.o : $(ddrApp)_shvdlib.combined
	$(ECHO) $(OBJCOPY) -I binary --rename-section .data=.ddr.data          \
    --redefine-sym  _binary_$(subst /,_,$(subst .,_,$<))_start=jumpTable   \
    -O elf32-sparc -B sparc $< $@

# -----------------------------[ Build Options ]------------------------------ #

ifeq (yes, $(PROFILE_MODE))
lrt_profile_build:
CCOPT       += -DPROFILE
CCOPT_LRT	+= -DPROFILE
endif
ifeq (yes, $(DUAL_CPU_MODE))
lrt_dual_exec_build:
CCOPT       += -DDUAL_CPU
CCOPT_LRT	+= -DDUAL_CPU
endif


CC_INCLUDE      += $(patsubst %, -I%, $(wildcard leon/Configuration))
MVCC_INCLUDE    += $(patsubst %, -I%, $(wildcard leon/Configuration))

CC_INCLUDE      += $(patsubst %, -I%, $(wildcard leon/Network))
MVCC_INCLUDE    += $(patsubst %, -I%, $(wildcard leon/Network))

CC_INCLUDE      += $(patsubst %, -I%, $(wildcard leon/Dma))
MVCC_INCLUDE    += $(patsubst %, -I%, $(wildcard leon/Dma))

CC_INCLUDE      += $(patsubst %, -I%, $(wildcard leon/Data))
MVCC_INCLUDE    += $(patsubst %, -I%, $(wildcard leon/Data))

LDDYNOPT =-L . -L ./scripts --nmagic -s
LDDYNOPT+=-T ./scripts/ld/lib.ldscript

LDSYMOPT =-L . -L ./scripts --nmagic
LDSYMOPT+=-T ./scripts/ld/lib.ldscript


# ------------------------------[ Extra Rules ]------------------------------- #

MV_BUILD_CONFIG ?= release

TEST_TYPE        := AUTO
TEST_TAGS        := "MA2150, TCL_MA2150, MA2450"
TEST_HW_PLATFORM := "MV0182_MA2150, MV0212_MA2450"