dnl HAVE_WARN_EFFECTIVE_CXX
dnl ----------------------
dnl
dnl If the C++ compiler accepts the `-Weffc++' flag,
dnl set output variable `WARN_EFFECTIVE_CXX' to `-Weffc++' and
dnl `WARN_NO_EFFECTIVE_CXX' to `-Wno-effc++'.  Otherwise,
dnl leave both empty.
dnl
AC_DEFUN([HAVE_WARN_EFFECTIVE_CXX],
[
AC_REQUIRE([AC_PROG_CXX])
AC_MSG_CHECKING([whether the C++ compiler (${CXX}) accepts -Weffc++])
AC_CACHE_VAL([cv_warn_effective_cxx],
[
AC_LANG_SAVE
AC_LANG_CPLUSPLUS
save_cxxflags="$CXXFLAGS"
CXXFLAGS="$CXXFLAGS -Weffc++"
AC_TRY_COMPILE([],[main();],
[cv_warn_effective_cxx=yes], [cv_warn_effective_cxx=no])
CXXFLAGS="$save_cxxflags"
AC_LANG_RESTORE
])
AC_MSG_RESULT([$cv_warn_effective_cxx])
if test "$cv_warn_effective_cxx" = yes; then
	WARN_EFFECTIVE_CXX=-Weffc++
	WARN_NO_EFFECTIVE_CXX=-Wno-effc++
fi
AC_SUBST([WARN_EFFECTIVE_CXX])
AC_SUBST([WARN_NO_EFFECTIVE_CXX])
])
