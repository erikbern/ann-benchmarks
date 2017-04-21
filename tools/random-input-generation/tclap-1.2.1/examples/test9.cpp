#include <string>
#include <iostream>
#include <algorithm>
#include "tclap/CmdLine.h"

using namespace TCLAP;
using namespace std;

int main(int argc, char** argv)
{
	try {

	CmdLine cmd("Command description message", ' ', "0.9",false);

	SwitchArg reverseSwitch("r","reverse","REVERSE instead of FORWARDS", false);
	cmd.add( reverseSwitch );

	MultiSwitchArg verbose("V","verbose","Level of verbosity");
	cmd.add( verbose );

	MultiSwitchArg noise("N","noise","Level of noise",5);
	cmd.add( noise );

	UnlabeledValueArg<string> word("word","a random word", false, "string",
					               "won't see this",false);
	cmd.add( word );

	// Uncommenting the next arg will (correctly) cause an exception
	// to be thrown.

//	UnlabeledMultiArg<string> badword("badword","a bad word", false,"string");
//
//	cmd.add( badword );

	cmd.parse( argc, argv );

	bool reverseName = reverseSwitch.getValue();

	if ( reverseName )
		cout << "REVERSE" << endl;
	else
		cout << "FORWARD" << endl;

	if ( verbose.isSet() )
		cout << "Verbose level: " << verbose.getValue() << endl;

	if ( noise.isSet() )
		cout << "Noise level: " << noise.getValue() << endl;

	if ( word.isSet() )
		cout << "Word: " << word.getValue() << endl;

	} catch (ArgException &e)  // catch any exceptions
	{ cerr << "error: " << e.error() << " for arg " << e.argId() << endl; }
}

