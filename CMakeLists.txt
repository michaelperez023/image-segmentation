cmake_minimum_required(VERSION 3.16)
project(Segmentation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
message(STATUS "OpenCV ${OpenCV_VERSION} found in ${OpenCV_DIR}")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra)
elseif(MSVC)
  add_compile_options(/W4)
endif()

# Otsu thresholding and mean shift segmentation
add_library(segmentation_core STATIC Otsu.cpp Otsu.h MeanShift.cpp MeanShift.h)
target_include_directories(segmentation_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(segmentation_core PUBLIC opencv_core opencv_imgproc)

# Command-line program
add_executable(segment main.cpp)
target_link_libraries(segment PRIVATE segmentation_core opencv_imgcodecs opencv_highgui)

option(BUILD_TESTING "Build the tests" ON)
if(BUILD_TESTING)
  enable_testing()
  set(IMAGES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/images)
  set(TEST_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_output)

  add_executable(otsu_tests tests/test_otsu.cpp tests/check.h)
  target_link_libraries(otsu_tests PRIVATE segmentation_core opencv_imgcodecs)
  foreach(name IN ITEMS four_levels brute_force sample_images repeat_calls opencv_otsu
                        class_boundaries per_channel degenerate)
    add_test(NAME otsu_${name} COMMAND otsu_tests ${name} ${IMAGES_DIR})
  endforeach()

  add_executable(meanshift_tests tests/test_meanshift.cpp tests/check.h)
  target_link_libraries(meanshift_tests PRIVATE segmentation_core opencv_imgcodecs)
  foreach(name IN ITEMS lab_conversion piecewise_constant colour_distance convergence symmetry
                        small_regions sample_images arguments)
    add_test(NAME meanshift_${name} COMMAND meanshift_tests ${name} ${IMAGES_DIR})
  endforeach()

  # The program itself, without a display, on every sample image. The error
  # cases are recognised by their messages.
  file(GLOB SAMPLE_IMAGES ${IMAGES_DIR}/*/*.png)
  foreach(image IN LISTS SAMPLE_IMAGES)
    get_filename_component(stem ${image} NAME_WE)
    string(REPLACE "-150x150" "" stem ${stem})
    add_test(NAME cli_${stem} COMMAND segment --no-gui --out ${TEST_OUTPUT} ${image})
  endforeach()
  set(SAMPLE_IMAGE ${IMAGES_DIR}/otsu4/blocks_L-150x150.png)
  add_test(NAME cli_one_method COMMAND segment --no-gui --method otsu4 --out ${TEST_OUTPUT}/one ${SAMPLE_IMAGE})
  set_tests_properties(cli_one_method PROPERTIES PASS_REGULAR_EXPRESSION "thresholds 72, 114, 159"
                       FAIL_REGULAR_EXPRESSION "Mean shift|per channel|Could not")
  add_test(NAME cli_otsu2 COMMAND segment --no-gui --method otsu2 --out ${TEST_OUTPUT}/otsu2 ${SAMPLE_IMAGE})
  set_tests_properties(cli_otsu2 PROPERTIES
                       PASS_REGULAR_EXPRESSION "threshold 109, eta 0.662, class sizes 7276, 15224"
                       FAIL_REGULAR_EXPRESSION "Mean shift|4 classes|Could not")
  add_test(NAME cli_missing_image COMMAND segment --no-gui --out ${TEST_OUTPUT} missing.png)
  add_test(NAME cli_bad_method COMMAND segment --no-gui --method semantic ${SAMPLE_IMAGE})
  add_test(NAME cli_bad_bandwidth COMMAND segment --no-gui --hs 0 ${SAMPLE_IMAGE})
  add_test(NAME cli_bad_min_region COMMAND segment --no-gui --min-region -5 ${SAMPLE_IMAGE})
  add_test(NAME cli_unknown_option COMMAND segment --no-gui --classes 3 ${SAMPLE_IMAGE})
  set_tests_properties(cli_missing_image PROPERTIES PASS_REGULAR_EXPRESSION "Could not read the image")
  set_tests_properties(cli_bad_method PROPERTIES
                       PASS_REGULAR_EXPRESSION "--method must be otsu2, otsu4, meanshift or all")
  set_tests_properties(cli_bad_bandwidth PROPERTIES PASS_REGULAR_EXPRESSION "--hs needs a number from 1 to 50")
  set_tests_properties(cli_bad_min_region PROPERTIES
                       PASS_REGULAR_EXPRESSION "--min-region needs an integer from 0 to 1000000")
  set_tests_properties(cli_unknown_option PROPERTIES PASS_REGULAR_EXPRESSION "Unknown option: --classes")

  # An output path that is a file must be reported before any method runs
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/not_a_directory "")
  add_test(NAME cli_bad_output_dir
           COMMAND segment --no-gui --out ${CMAKE_CURRENT_BINARY_DIR}/not_a_directory ${SAMPLE_IMAGE})
  set_tests_properties(cli_bad_output_dir PROPERTIES PASS_REGULAR_EXPRESSION "Could not create"
                       FAIL_REGULAR_EXPRESSION "Otsu")
endif()
