cmake_minimum_required(VERSION 3.5)

project(Desevi VERSION 0.1 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt5 COMPONENTS Widgets REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets REQUIRED)

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
file(GLOB_RECURSE UIS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.ui)

set(PROJECT_SOURCES
        main.cpp
        ${SOURCES}
        ${HEADERS}
        ${UIS}
)

add_executable(Desevi ${PROJECT_SOURCES})
target_include_directories(Desevi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(Desevi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/external)


target_link_libraries(Desevi PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)
