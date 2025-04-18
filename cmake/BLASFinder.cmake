# Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

message( DEBUG "BLAS_LIBRARIES '${BLAS_LIBRARIES}'"        )
message( DEBUG "  cached       '${blas_libraries_cached}'" )
message( DEBUG "blas           '${blas}'"                  )
message( DEBUG "  cached       '${blas_cached}'"           )
message( DEBUG "blas_int       '${blas_int}'"              )
message( DEBUG "  cached       '${blas_int_cached}'"       )
message( DEBUG "blas_fortran   '${blas_fortran}'"          )
message( DEBUG "  cached       '${blas_fortran_cached}'"   )
message( DEBUG "blas_threaded  '${blas_threaded}'"         )
message( DEBUG "  cached       '${blas_threaded_cached}'"  )
message( DEBUG "" )

include( "cmake/util.cmake" )

#-----------------------------------
# Check if this file has already been run with these settings (see bottom).
if (BLAS_LIBRARIES)
    # At this point, BLAS_LIBRARIES comes from CMake FindBLAS or
    # user input (`cmake -DBLAS_LIBRARIES=...`).
    if (BLAS_LIBRARIES STREQUAL blas_libraries_cached)
        # Already checked this BLAS_LIBRARIES; load cached results.
        message( STATUS "Using cached BLAS_LIBRARIES settings" )
        set( BLAS_FOUND "${blas_found_cached}" )
        set( run_ false )
    else()
        # Need to check BLAS_LIBRARIES.
        set( run_ true )
        # Clear blas so test_mkl, etc. later are false.
        set( blas "" )
    endif()

elseif (    "${blas}"          STREQUAL "${blas_cached}"
        AND "${blas_fortran}"  STREQUAL "${blas_fortran_cached}"
        AND "${blas_int}"      STREQUAL "${blas_int_cached}"
        AND "${blas_threaded}" STREQUAL "${blas_threaded_cached}")
    # Already checked this blas; load cached results.
    message( STATUS "Using cached blas settings" )
    set( BLAS_LIBRARIES "${blas_libraries_cached}" )
    set( BLAS_FOUND     "${blas_found_cached}" )
    set( run_ false )
else()
    # Search blas, blas_int, etc.
    set( run_ true )
endif()

#===============================================================================
# Matching endif at bottom.
if (run_)

#-------------------------------------------------------------------------------
# Prints the BLAS_{name,libs}_lists.
# This uses CMAKE_MESSAGE_LOG_LEVEL rather than message( DEBUG, ... )
# because the extra "-- " cmake prints were quite distracting.
# Usage: cmake -DCMAKE_MESSAGE_LOG_LEVEL=DEBUG ..
#
function( debug_print_list msg )
    if ("${CMAKE_MESSAGE_LOG_LEVEL}" MATCHES "DEBUG|TRACE")
        message( "---------- lists: ${msg}" )
        message( "blas_name_list = ${blas_name_list}" )
        message( "blas_libs_list = ${blas_libs_list}" )

        message( "\nrow;  ${red}blas_name;${plain}  blas_libs" )
        set( i 0 )
        foreach (name IN LISTS blas_name_list)
            list( GET blas_libs_list ${i} libs )
            message( "${i};  ${red}${name};${plain}  ${libs}" )
            math( EXPR i "${i} + 1" )
        endforeach()
        message( "" )
    endif()
endfunction()

#-------------------------------------------------------------------------------
# Setup.

#---------------------------------------- compiler
string( COMPARE EQUAL "${CMAKE_CXX_COMPILER_ID}" "GNU"        gnu_compiler)
string( COMPARE EQUAL "${CMAKE_CXX_COMPILER_ID}" "IntelLLVM"  intelllvm_compiler )
string( COMPARE EQUAL "${CMAKE_CXX_COMPILER_ID}" "Intel"      intel_compiler )
match( "XL|XLClang" "${CMAKE_CXX_COMPILER_ID}" ibm_compiler )

#---------------------------------------- Fortran manglings to test
if (ibm_compiler)
    # For IBM XL, change default mangling search order to lower, add_, upper,
    # ESSL includes all 3, but Netlib LAPACK has only one mangling.
    set( fortran_mangling_list
        "-DBLAS_FORTRAN_LOWER"
        "-DBLAS_FORTRAN_ADD_"
        "-DBLAS_FORTRAN_UPPER"
    )
else()
    # For all others, mangling search order as add_, lower, upper,
    # since add_ is the most common.
    set( fortran_mangling_list
        "-DBLAS_FORTRAN_ADD_"
        "-DBLAS_FORTRAN_LOWER"
        "-DBLAS_FORTRAN_UPPER"
    )
endif()
message( DEBUG "fortran_mangling_list = ${fortran_mangling_list}" )

#-------------------------------------------------------------------------------
# Parse options: BLAS_LIBRARIES, blas, blas_int, blas_threaded, blas_fortran.

#---------------------------------------- blas
string( TOLOWER "${blas}" blas_ )

match( "auto|apple|accelerate"    "${blas_}" test_accelerate )
match( "auto|aocl|blis"           "${blas_}" test_blis       )
match( "auto|cray|libsci|default" "${blas_}" test_default    )
match( "auto|ibm|essl"            "${blas_}" test_essl       )
match( "auto|intel|mkl"           "${blas_}" test_mkl        )
match( "auto|openblas"            "${blas_}" test_openblas   )
match( "auto|generic"             "${blas_}" test_generic    )

message( DEBUG "
BLAS_LIBRARIES      = '${BLAS_LIBRARIES}'
blas                = '${blas}'
blas_               = '${blas_}'
test_accelerate     = '${test_accelerate}'
test_blis           = '${test_blis}'
test_default        = '${test_default}'
test_essl           = '${test_essl}'
test_mkl            = '${test_mkl}'
test_openblas       = '${test_openblas}'
test_generic        = '${test_generic}'" )

#---------------------------------------- blas_fortran
string( TOLOWER "${blas_fortran}" blas_fortran_ )

match( "auto|gfortran" "${blas_fortran_}" test_gfortran )
match( "auto|ifort"    "${blas_fortran_}" test_ifort    )

message( DEBUG "
blas_fortran        = '${blas_fortran}'
blas_fortran_       = '${blas_fortran_}'
test_gfortran       = '${test_gfortran}'
test_ifort          = '${test_ifort}'")

#---------------------------------------- blas_int
string( TOLOWER "${blas_int}" blas_int_ )

# This regex is similar to "\b(lp64|int)\b".
set( regex_int32 "(^|[^a-zA-Z0-9_])(auto|lp64|int|int32|int32_t)($|[^a-zA-Z0-9_])" )
set( regex_int64 "(^|[^a-zA-Z0-9_])(auto|ilp64|int64|int64_t)($|[^a-zA-Z0-9_])" )
match( "${regex_int32}" "${blas_int_}" test_int   )
match( "${regex_int64}" "${blas_int_}" test_int64 )
if (NOT (test_int OR test_int64))
    message( FATAL_ERROR,
             "Expected at least one of test_int=${test_int}"
             " or test_int64=${test_int64} to be true." )
endif()

#---------------------------------------- integer sizes to test
# blas_int, above, filters which libraries to test, e.g., mkl_lp64 or mkl_ilp64.
# After filtering, regardless of blas_int, we usually test all libraries
# with int32, and if that fails, with int64. With the current test,
# an int64 library will fail with blas_int=int32 and pass with blas_int=int64,
# an int32 library will pass with blas_int=int32 and pass (erroneously)
# with blas_int=int64. However, for cross compiling, we must rely on the
# user setting blas_int correctly.

set( int_size_list
    " "             # int32 (LP64)
    "-DBLAS_ILP64"  # int64 (ILP64)
)

if (CMAKE_CROSSCOMPILING)
    if (test_int AND test_int64)
        message( FATAL_ERROR " ${red}When cross-compiling, one must define either\n"
                 " `blas_int=int32` (usual convention) xor\n"
                 " `blas_int=int64` (ilp64 convention).${plain}" )
    elseif (test_int)
        list( POP_BACK  int_size_list tmp )  # remove int64 entry
    elseif (test_int64)
        list( POP_FRONT int_size_list tmp )  # remove int32 entry
    endif()
endif()

message( DEBUG "
blas_int            = '${blas_int}'
blas_int_           = '${blas_int_}'
test_int            = '${test_int}'
test_int64          = '${test_int64}'
int_size_list       = '${int_size_list}'")

#---------------------------------------- blas_threaded
string( TOLOWER "${blas_threaded}" blas_threaded_ )

# These regex are similar to "\b(yes|...)\b".
# All sequential BLAS also act sequentially inside OpenMP parallel sections,
# so `openmp_aware` sets test_sequential = true.
set( regex_thr "(^|[^a-zA-Z0-9_])(auto|y|yes|true|on|1)($|[^a-zA-Z0-9_])" )
set( regex_seq "(^|[^a-zA-Z0-9_])(auto|n|no|false|off|0|openmp_aware)($|[^a-zA-Z0-9_])" )
match( "${regex_thr}" "${blas_threaded_}" test_threaded )
match( "${regex_seq}" "${blas_threaded_}" test_sequential )
match( "openmp_aware" "${blas_threaded_}" test_threaded_omp )
if (NOT (test_threaded OR test_sequential))
    message( FATAL_ERROR,
             "Expected at least one of test_threaded=${test_threaded}"
             " or test_sequential=${test_sequential} to be true." )
endif()

message( DEBUG "
blas_threaded       = '${blas_threaded}'
blas_threaded_      = '${blas_threaded_}'
test_threaded       = '${test_threaded}'
test_sequential     = '${test_sequential}'
test_threaded_omp   = '${test_threaded_omp}'")

#-------------------------------------------------------------------------------
# Build list of libraries to check.
# todo: BLAS_?(ROOT|DIR)

set( blas_name_list "" )
set( blas_libs_list "" )

#---------------------------------------- BLAS_LIBRARIES
if (BLAS_LIBRARIES)
    # Change ; semi-colons to spaces so we can append it as one item to a list.
    string( REPLACE ";" " " BLAS_LIBRARIES_ESC "${BLAS_LIBRARIES}" )
    message( DEBUG "BLAS_LIBRARIES ${BLAS_LIBRARIES}" )
    message( DEBUG "   =>          ${BLAS_LIBRARIES_ESC}" )

    list( APPEND blas_name_list "BLAS_LIBRARIES" )
    list( APPEND blas_libs_list "${BLAS_LIBRARIES_ESC}" )
    debug_print_list( "BLAS_LIBRARIES" )
endif()

#---------------------------------------- default; Cray libsci
if (test_default)
    list( APPEND blas_name_list "default (no library)" )
    list( APPEND blas_libs_list " " )  # Use space so APPEND works later.
    debug_print_list( "default" )
endif()

#---------------------------------------- Intel MKL
# MKL is OpenMP aware: inside an application's OpenMP section,
# MKL does not open a new OpenMP section and spawn new threads, by default.
# See MKL_DYNAMIC.
# (I don't think this was always true, but has been true for several years now.)
if (test_mkl)
    # todo: MKL_?(ROOT|DIR)
    if ((test_threaded OR test_threaded_omp) AND OpenMP_CXX_FOUND)
        if (test_gfortran AND gnu_compiler)
            # GNU compiler + OpenMP: require gnu_thread library.
            if (test_int)
                list( APPEND blas_name_list "Intel MKL lp64,  GNU threads (gomp), gfortran")
                list( APPEND blas_libs_list "-lmkl_gf_lp64  -lmkl_gnu_thread -lmkl_core" )
            endif()

            if (test_int64)
                list( APPEND blas_name_list "Intel MKL ilp64, GNU threads (gomp), gfortran")
                list( APPEND blas_libs_list "-lmkl_gf_ilp64 -lmkl_gnu_thread -lmkl_core" )
            endif()

        elseif (test_ifort AND intelllvm_compiler)
            # IntelLLVM compiler + OpenMP: require intel_thread library.
            if (test_int)
                list( APPEND blas_name_list "Intel MKL lp64,  Intel threads (iomp5), ifort")
                list( APPEND blas_libs_list "-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core" )
            elseif (test_int64)
                list( APPEND blas_name_list "Intel MKL ilp64, Intel threads (iomp5), ifort")
                list( APPEND blas_libs_list "-lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core" )
            endif()

        elseif (test_ifort AND intel_compiler)
            # Intel compiler + OpenMP: require intel_thread library.
            if (test_int)
                list( APPEND blas_name_list "Intel MKL lp64,  Intel threads (iomp5), ifort")
                list( APPEND blas_libs_list "-lmkl_intel_lp64  -lmkl_intel_thread -lmkl_core" )
            endif()

            if (test_int64)
                list( APPEND blas_name_list "Intel MKL ilp64, Intel threads (iomp5), ifort")
                list( APPEND blas_libs_list "-lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core" )
            endif()

        else()
            # MKL doesn't have libraries for other OpenMP backends.
            message( "Skipping threaded MKL for non-GNU, non-Intel compiler with OpenMP" )
        endif()
    endif()

    #----------
    if (test_sequential)
        # If Intel compiler, prefer Intel ifort interfaces.
        if (test_ifort AND intel_compiler)
            if (test_int)
                list( APPEND blas_name_list "Intel MKL lp64,  sequential, ifort" )
                list( APPEND blas_libs_list "-lmkl_intel_lp64  -lmkl_sequential -lmkl_core" )
            endif()

            if (test_int64)
                list( APPEND blas_name_list "Intel MKL ilp64, sequential, ifort" )
                list( APPEND blas_libs_list "-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core" )
            endif()
        endif()  # ifort

        # Otherwise, prefer GNU gfortran interfaces.
        if (test_gfortran)
            if (test_int)
                list( APPEND blas_name_list "Intel MKL lp64,  sequential, gfortran" )
                list( APPEND blas_libs_list "-lmkl_gf_lp64  -lmkl_sequential -lmkl_core" )
            endif()

            if (test_int64)
                list( APPEND blas_name_list "Intel MKL ilp64, sequential, gfortran" )
                list( APPEND blas_libs_list "-lmkl_gf_ilp64 -lmkl_sequential -lmkl_core" )
            endif()
        endif()  # gfortran

        # Not Intel compiler, lower preference for Intel ifort interfaces.
        # todo: same as Intel block above.
        if (test_ifort AND NOT intel_compiler)
            if (test_int)
                list( APPEND blas_name_list "Intel MKL lp64,  sequential, ifort" )
                list( APPEND blas_libs_list "-lmkl_intel_lp64  -lmkl_sequential -lmkl_core" )
            endif()

            if (test_int64)
                list( APPEND blas_name_list "Intel MKL ilp64, sequential, ifort" )
                list( APPEND blas_libs_list "-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core" )
            endif()
        endif()  # ifort && not intel
    endif()  # sequential
    debug_print_list( "mkl" )
endif()  # MKL

#---------------------------------------- IBM ESSL
if (test_essl)
    # todo: ESSL_?(ROOT|DIR)
    if (test_threaded)
        #message( "essl OpenMP_CXX_FOUND ${OpenMP_CXX_FOUND}" )
        #if (ibm_compiler)
        #    if (test_int)
        #        list( APPEND blas_name_list "IBM ESSL int (lp64), multi-threaded"  )
        #        list( APPEND blas_libs_list "-lesslsmp -lxlsmp"  )
        #        # ESSL manual says '-lxlf90_r -lxlfmath' also,
        #        # but this doesn't work on Summit
        #    endif()
        #
        #    if (test_int64)
        #        list( APPEND blas_name_list "IBM ESSL int64 (ilp64), multi-threaded"  )
        #        list( APPEND blas_libs_list "-lesslsmp6464 -lxlsmp"  )
        #    endif()
        #else
        if (OpenMP_CXX_FOUND)
            if (test_int)
                list( APPEND blas_name_list "IBM ESSL int (lp64), multi-threaded, with OpenMP"  )
                list( APPEND blas_libs_list "-lesslsmp"  )
            endif()

            if (test_int64)
                list( APPEND blas_name_list "IBM ESSL int64 (ilp64), multi-threaded, with OpenMP"  )
                list( APPEND blas_libs_list "-lesslsmp6464"  )
            endif()
        endif()
    endif()  # threaded

    if (test_sequential)
        if (test_int)
            list( APPEND blas_name_list "IBM ESSL int (lp64), sequential"  )
            list( APPEND blas_libs_list "-lessl"  )
        endif()

        if (test_int64)
            list( APPEND blas_name_list "IBM ESSL int64 (ilp64), sequential"  )
            list( APPEND blas_libs_list "-lessl6464"  )
        endif()
    endif()  # sequential
    debug_print_list( "essl" )
endif()

#---------------------------------------- OpenBLAS
if (test_openblas)
    # todo: OPENBLAS_?(ROOT|DIR)
    list( APPEND blas_name_list "OpenBLAS" )
    list( APPEND blas_libs_list "-lopenblas" )
    debug_print_list( "openblas" )
endif()

#---------------------------------------- BLIS (also used by AMD AOCL)
if (test_blis)
    if (test_threaded)
        list( APPEND blas_name_list "BLIS and FLAME, multi-threaded" )
        list( APPEND blas_libs_list "-lflame -lblis-mt" )
    endif()
    if (test_sequential)
        list( APPEND blas_name_list "BLIS and FLAME" )
        list( APPEND blas_libs_list "-lflame -lblis" )
    endif()
    debug_print_list( "blis" )
endif()

#---------------------------------------- Apple Accelerate
if (test_accelerate)
    list( APPEND blas_name_list "Apple Accelerate" )
    list( APPEND blas_libs_list "-framework Accelerate" )
    debug_print_list( "accelerate" )
endif()

#---------------------------------------- generic -lblas
if (test_generic)
    list( APPEND blas_name_list "generic" )
    list( APPEND blas_libs_list "-lblas" )
    debug_print_list( "generic" )
endif()

#-------------------------------------------------------------------------------
# Check each BLAS library.

# Reset CMake's FindBLAS status. Consider BLAS found if we can link and
# run with it below.
set( BLAS_FOUND false )
unset( blaspp_defs_ CACHE )

set( i 0 )
foreach (blas_name IN LISTS blas_name_list)
    message( TRACE "i: ${i}" )
    list( GET blas_libs_list ${i} blas_libs )
    math( EXPR i "${i}+1" )

    if (i GREATER 1)
        message( "" )
    endif()
    message( "${blas_name}" )
    message( "   libs:  ${blas_libs}" )

    # Strip to deal with default lib being space, " ".
    # Split on spaces to make list,
    # but keep '-framework Accelerate' together as one item.
    message( DEBUG "   blas_libs: '${blas_libs}'" )
    string( STRIP "${blas_libs}" blas_libs )
    string( REGEX REPLACE " +" ";" blas_libs "${blas_libs}" )
    string( REGEX REPLACE "-framework;" "-framework " blas_libs "${blas_libs}" )
    message( DEBUG "   blas_libs: '${blas_libs}' (split)" )

    foreach (mangling IN LISTS fortran_mangling_list)
        foreach (int_size IN LISTS int_size_list)
            set( label "   ${mangling} ${int_size}" )
            pad_string( "${label}" 50 label )

            # Try to link a simple hello world with the library.
            try_compile(
                link_result ${CMAKE_CURRENT_BINARY_DIR}
                SOURCES
                    "${CMAKE_CURRENT_SOURCE_DIR}/config/hello.cc"
                LINK_LIBRARIES
                    ${blas_libs} ${openmp_lib} # not "..." quoted; screws up OpenMP
                COMPILE_DEFINITIONS
                    "${mangling} ${int_size}"
                OUTPUT_VARIABLE
                    link_output
            )
            debug_try_compile( "hello.cc" "${link_result}" "${link_output}" )

            # If hello didn't link, assume library not found,
            # so break both mangling & int_size loops.
            if (NOT link_result)
                message( "${label} ${red} no (library not found)${plain}" )
                break()
            endif()

            # Try to link and run simple BLAS routine with the library.
            try_run(
                run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
                SOURCES
                    "${CMAKE_CURRENT_SOURCE_DIR}/config/blas.cc"
                LINK_LIBRARIES
                    ${blas_libs} ${openmp_lib} # not "..." quoted; screws up OpenMP
                COMPILE_DEFINITIONS
                    "${mangling} ${int_size}"
                COMPILE_OUTPUT_VARIABLE
                    compile_output
                RUN_OUTPUT_VARIABLE
                    run_output
            )
            # For cross-compiling, if it links, assume the run is okay.
            # User must set blas_int=int64 for ILP64, otherwise assumes int32.
            if (CMAKE_CROSSCOMPILING AND compile_result)
                message( DEBUG "cross: blas_int = '${blas_int}'" )
                set( run_result "0"  CACHE STRING "" FORCE )
                set( run_output "ok" CACHE STRING "" FORCE )
            endif()
            debug_try_run( "blas.cc" "${compile_result}" "${compile_output}"
                                     "${run_result}" "${run_output}" )

            if (NOT compile_result)
                # If int32 didn't link, int64 won't either, so break int_size loop.
                message( "${label} ${red} no (didn't link: routine not found)${plain}" )
                break()
            elseif ("${run_output}" MATCHES "ok")
                # If it runs and prints ok, we're done, so break all 3 loops.
                message( "${label} ${blue} yes${plain}" )

                set( BLAS_FOUND true )
                if (BLAS_LIBRARIES)
                    # BLAS_LIBRARIES from CMake FindBLAS or user input
                    # shouldn't get changed, except being split into list.
                    string( REPLACE " " ";" blas_libraries_ "${BLAS_LIBRARIES}" )
                    if (NOT blas_libraries_ STREQUAL blas_libs)
                        message( WARNING "Expected BLAS_LIBRARIES = '${BLAS_LIBRARIES}'\n"
                                         "to match blas_libs      = '${blas_libs}'" )
                    endif()
                else()
                    set( BLAS_LIBRARIES "${blas_libs}" )
                endif()

                if (mangling MATCHES "[^ ]")  # non-empty
                    list( APPEND blaspp_defs_ "${mangling}" )
                endif()
                if (int_size MATCHES "[^ ]")  # non-empty
                    list( APPEND blaspp_defs_ "${int_size}" )
                endif()
                if (int_size MATCHES "ILP64")
                    set( blaspp_int "int64" )
                else()
                    set( blaspp_int "int32" )
                endif()
                break()
            else()
                message( "${label} ${red} no (didn't run: int mismatch, etc.)${plain}" )
            endif()
        endforeach()  # int_size

        # Break loops as described above.
        if (NOT link_result OR BLAS_FOUND)
            break()
        endif()
    endforeach()  # mangling

    # Break loops as described above.
    if (BLAS_FOUND)
        break()
    endif()
endforeach()  # blas_name

# Mark as already run (see top).
set( blas_libraries_cached "${BLAS_LIBRARIES}" CACHE INTERNAL "" )
set( blas_found_cached     "${BLAS_FOUND}"     CACHE INTERNAL "" )
set( blas_cached           "${blas}"           CACHE INTERNAL "" )
set( blas_fortran_cached   "${blas_fortran}"   CACHE INTERNAL "" )
set( blas_int_cached       "${blas_int}"       CACHE INTERNAL "" )
set( blas_threaded_cached  "${blas_threaded}"  CACHE INTERNAL "" )

endif() # run_
#===============================================================================


#-------------------------------------------------------------------------------
if (BLAS_FOUND)
    message( "${blue}   Found BLAS library: ${BLAS_LIBRARIES}${plain}" )
else()
    message( "${red}   BLAS library not found.${plain}" )
endif()

message( DEBUG "
BLAS_FOUND          = '${BLAS_FOUND}'
BLAS_LIBRARIES      = '${BLAS_LIBRARIES}'
blaspp_defs_        = '${blaspp_defs_}'
")
message( "" )
