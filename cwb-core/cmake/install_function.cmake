#Get the full path to the script dir
get_filename_component(CWB_SCRIPTS_DIR
                       ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_SYSCONFDIR}/cwb/scripts/
                       ABSOLUTE
                       BASE_DIR ${CMAKE_BINARY_DIR})

function (install_files)
    if (UNIX)
        foreach (file ${ARGV})
            get_filename_component (name_without_extension ${file} NAME_WE)
            get_filename_component (name_without_directory ${file} NAME)
            get_filename_component (extension ${file} LAST_EXT)
            install (FILES ${file}
                   DESTINATION ${CWB_SCRIPTS_DIR}
                   PERMISSIONS OWNER_WRITE
                               OWNER_READ GROUP_READ WORLD_READ
                               OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE)
            if (NOT "${extension}" STREQUAL ".py")
                install(
                  CODE "execute_process( \
                    COMMAND ${CMAKE_COMMAND} -E create_symlink   \
                        ${CWB_SCRIPTS_DIR}/${name_without_directory} \
                        ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/${name_without_extension}  \
                  )"
                )
                install(
                    CODE "message(\"-- Created symlink: ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/${name_without_extension} -> \
                    ${CWB_SCRIPTS_DIR}/${name_without_directory}\")
                ")
            endif ()
        endforeach ()
    else (WIN32)
        install (FILES ${ARGV} DESTINATION bin)
    endif ()
endfunction ()
