AC_DEFUN([BB_ENABLE_DOXYGEN],
[
AC_ARG_ENABLE(doxygen, [--enable-doxygen 	enable documentation generation with doxygen (auto)])
if test "x$enable_doxygen" = xno; then
        enable_doc=no
else 
        AC_PATH_PROG(DOXYGEN, doxygen, , $PATH)
        if test x$DOXYGEN = x; then
                if test "x$enable_doxygen" = xyes; then
                        AC_MSG_ERROR([could not find doxygen])
                fi
                enable_doc=no
        else
                enable_doc=yes
        fi
fi
AM_CONDITIONAL(DOC, test x$enable_doc = xyes)
])
