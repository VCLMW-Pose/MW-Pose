cmake_minimum_required(VERSION 3.12)
project(demo)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")

set(OpenCV_DIR "D:\\opencv\\mingw64_build\\install")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_path(WALABOT_INCLUDE_DIR WalabotAPI.h  PATHS C:\\Program\ Files\\Walabot\\WalabotSDK\\inc)
find_library(WALABOT_LIB WalabotAPI  PATHS C:\\Program\ Files\\Walabot\\WalabotSDK\\bin)
find_file(WALABOT_DLL WalabotAPI.dll PATHS C:\\Program\ Files\\Walabot\\WalabotSDK\\bin)
set(LIBUSB_DLL C:\\Program\ Files\\Walabot\\WalabotSDK\\bin\\libusb-1.0.dll)
set(QT_DLL C:\\Program\ Files\\Walabot\\WalabotSDK\\bin\\Qt5Core.dll)
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

include_directories(${WALABOT_INCLUDE_DIR})

add_executable(demo main.cpp walabot.cpp walabot.hpp walabot_test.cpp)
target_link_libraries(demo ${OpenCV_LIBS})

foreach (TARGET_NAME demo)

    target_link_libraries(${TARGET_NAME} ${WALABOT_LIB})

    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${WALABOT_DLL} ${LIBUSB_DLL} ${QT_DLL}
            $<TARGET_FILE_DIR:${TARGET_NAME}>)
endforeach()
