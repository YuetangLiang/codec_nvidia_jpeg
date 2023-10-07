project(nvjpg C CXX CUDA)

set(PUBLIC_DOCS
    README.md
)

set(SOURCES
  #nvjpeg_coder.cu
)

set(LIBRARIES
    ${LIBRARIES}
    nvjpeg
)

#add_executable(${PROJECT_NAME}_test ${SOURCES} test.cu)
#target_link_libraries(${PROJECT_NAME}_test PRIVATE ${LIBRARIES})
#add_library(_${PROJECT_NAME} SHARED ${SOURCES})
