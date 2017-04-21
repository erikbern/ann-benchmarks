// Test only makes sure we can use different argv types for the
// parser. Don't run, just compile.

#include "tclap/CmdLine.h"

using namespace TCLAP;
int main()
{
        char *argv5[] = {(char*)"Foo", 0};
	const char *argv6[] = {"Foo", 0};
	const char * const argv7[] = {"Foo", 0};
	char **argv1 = argv5;
	const char **argv2 = argv6;
	const char * const * argv3 = argv7;
	const char * const * const argv4 = argv7;

	CmdLine cmd("Command description message", ' ', "0.9");
	cmd.parse(0, argv1);
	cmd.parse(0, argv2);
	cmd.parse(0, argv3);
	cmd.parse(0, argv4);
	cmd.parse(0, argv5);
	cmd.parse(0, argv6);
	cmd.parse(0, argv7);
}

