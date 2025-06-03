cmake_minimum_required(VERSION 3.10)
project(VW_PnP_Localization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(main main.cpp)
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)

# nlohmann_json 라이브러리 포함 (header-only)
include(FetchContent)
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)
target_link_libraries(main PRIVATE nlohmann_json::nlohmann_json ${OpenCV_LIBS})
