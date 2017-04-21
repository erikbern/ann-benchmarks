dnl @synopsis AC_CXX_HAVE_SSTREAM
dnl
dnl If the C++ library has a working stringstream, define HAVE_SSTREAM.
dnl
dnl @author Ben Stanley
dnl @version $Id: ac_cxx_have_sstream.m4,v 1.2 2006/02/22 02:10:28 zeekec Exp $
dnl
AC_DEFUN([AC_CXX_HAVE_SSTREAM],
[AC_REQUIRE([AC_CXX_NAMESPACES])
AC_LANG_SAVE
AC_LANG_CPLUSPLUS
AC_CHECK_HEADERS(sstream)
AC_CACHE_CHECK([whether the STL defines stringstream],
[ac_cv_cxx_have_sstream],
[AC_TRY_COMPILE([#include <sstream>
#ifdef HAVE_NAMESPACES
using namespace std;
#endif],[stringstream message; message << "Hello"; return 0;],
 ac_cv_cxx_have_sstream=yes, ac_cv_cxx_have_sstream=no)
])
if test "$ac_cv_cxx_have_sstream" = yes; then
  AC_DEFINE(HAVE_SSTREAM,1,[define if the compiler has stringstream])
fi
AC_LANG_RESTORE
])
