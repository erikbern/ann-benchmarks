#include <string>
#include <iostream>
#include <algorithm>
#include "tclap/CmdLine.h"

using namespace TCLAP;
using namespace std;

int main(int argc, char** argv)
{
	try {

		CmdLine cmd("Command description message", ' ', "0.9", true);

		cmd.setExceptionHandling(false);

		cmd.parse(argc, argv);

	} catch (ArgException &e) { // catch any exceptions
		cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
		return 1;
	} catch (ExitException &e) { // catch any exceptions
		cerr << "Exiting on ExitException." << endl;
		return e.getExitStatus();
	}
}

