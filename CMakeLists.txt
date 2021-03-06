cmake_minimum_required(VERSION 3.5)

project(mlir-interactive VERSION 0.1 LANGUAGES CXX C)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# If on linux and using clang, force lld.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_LINK_FLAGS "-fuse-ld=lld")
endif()

# Generate a CompilationDatabase (compile_commands.json file) for our build,
# for use by clang_complete, YouCompleteMe, etc.
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

#-------------------------------------------------------------------------------
# MLIR/LLVM Configuration
#-------------------------------------------------------------------------------

find_package(CIRCT REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${CIRCT_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

set(MLIR_MAIN_SRC_DIR ${LLVM_BUILD_MAIN_SRC_DIR}/../mlir ) # --src-root
set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include ) # --includedir
set(MLIR_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CIRCT_INCLUDE_DIRS})

#-------------------------------------------------------------------------------
# mlir-interactive setup
#-------------------------------------------------------------------------------

find_package(QT NAMES Qt5 COMPONENTS Widgets REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets REQUIRED)

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
file(GLOB_RECURSE UIS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.ui)
file(GLOB_RECURSE RESOURCES ${CMAKE_CURRENT_SOURCE_DIR}/resources/*.qrc)

set(PROJECT_SOURCES
        main.cpp
        ${SOURCES}
        ${HEADERS}
        ${UIS}
        ${RESOURCES}
)

# set LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING in preprocessor
# to disable LLVM ABI breaking checks in headers.
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING")

# set QT_NO_EMIT to avoid colliding names in MLIR
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DQT_NO_EMIT")

add_executable(mlir-interactive ${PROJECT_SOURCES})
target_include_directories(mlir-interactive PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(mlir-interactive PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/external)
set(LLVM_REQUIRES_EH ON)
set(LLVM_REQUIRES_RTTI ON)
llvm_update_compile_flags(mlir-interactive)

# Qt links
target_link_libraries(mlir-interactive PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)

# LLVM/MLIR/CIRCT links
target_link_libraries(mlir-interactive
       PRIVATE
       LLVMSupport
       MLIRParser
       MLIRSideEffectInterfaces
       MLIRSupport
       MLIRTransforms

       # MLIR dialects
       MLIRIR
       MLIRLLVMIR
       MLIRMemRef
       MLIROptLib
       MLIRParser
       MLIRFunc
       MLIRSCF

       # MLIR conversions
       MLIRSCFToControlFlow

       # CIRCT dialects
       CIRCTCalyx
       CIRCTHandshake
       CIRCTFIRRTL
       CIRCTComb
       CIRCTHW
       CIRCTSeq
       CIRCTESI
       CIRCTFSM
       CIRCTLLHD
       CIRCTMSFT
       CIRCTMoore
       CIRCTStaticLogicOps

       # CIRCT conversions
       CIRCTStandardToHandshake
       CIRCTHandshakeToFIRRTL
)

#-------------------------------------------------------------------------------
# External setup
#-------------------------------------------------------------------------------

add_subdirectory(external)
target_include_directories(mlir-interactive PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/external/cereal/include)
