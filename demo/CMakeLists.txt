cmake_minimum_required(VERSION 3.12)
project(demo)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")

find_path(WALABOT_INCLUDE_DIR WalabotAPI.h  PATHS C:\\Program\ Files\\Walabot\\WalabotSDK\\inc)
find_library(WALABOT_LIB WalabotAPI  PATHS C:\\Program\ Files\\Walabot\\WalabotSDK\\bin)
find_file(WALABOT_DLL WalabotAPI.dll PATHS C:\\Program\ Files\\Walabot\\WalabotSDK\\bin)
set(LIBUSB_DLL C:\\Program\ Files\\Walabot\\WalabotSDK\\bin\\libusb-1.0.dll)
set(QT_DLL C:\\Program\ Files\\Walabot\\WalabotSDK\\bin\\Qt5Core.dll)

include_directories(${WALABOT_INCLUDE_DIR})

add_executable(demo main.cpp walabot.hpp)

target_link_libraries(demo ${WALABOT_LIB})