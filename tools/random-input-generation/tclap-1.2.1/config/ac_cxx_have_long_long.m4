dnl @synopsis AC_CXX_HAVE_LONG_LONG
dnl
dnl If the C++ implementation have a long long type
dnl
AC_DEFUN([AC_CXX_HAVE_LONG_LONG],
[AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_COMPILE([],[long long x = 1; return 0;],
 ac_cv_cxx_have_long_long=yes, ac_cv_cxx_have_long_long=no)

if test "$ac_cv_cxx_have_long_long" = yes; then
  AC_DEFINE(HAVE_LONG_LONG, 1,
  [define if the C++ implementation have long long])
else
  AC_DEFINE(HAVE_LONG_LONG, 0,
  [define if the C++ implementation have long long])
fi
AC_LANG_RESTORE
])
