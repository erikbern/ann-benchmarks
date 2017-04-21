#include <string>
#include "tclap/CmdLine.h"

using namespace TCLAP;
using namespace std;

int main(int argc, char** argv)
{
	// Wrap everything in a try block.  Do this every time,
	// because exceptions will be thrown for problems.
	try {

	// Define the command line object.
	CmdLine cmd("Command description message", ' ', "0.9");

	vector<string> allowed;
	allowed.push_back("homer");
	allowed.push_back("marge");
	allowed.push_back("bart");
	allowed.push_back("lisa");
	allowed.push_back("maggie");
	ValuesConstraint<string> allowedVals( allowed );

	ValueArg<string> nameArg("n","name","Name to print",true,"homer",
					         &allowedVals);
	cmd.add( nameArg );

	vector<int> iallowed;
	iallowed.push_back(1);
	iallowed.push_back(2);
	iallowed.push_back(3);
	ValuesConstraint<int> iallowedVals( iallowed );

	UnlabeledValueArg<int> intArg("times","Number of times to print",true,1,
					      &iallowedVals,false);
	cmd.add( intArg );

	// Parse the args.
	cmd.parse( argc, argv );

	// Get the value parsed by each arg.
	int num = intArg.getValue();
	string name = nameArg.getValue();

	for ( int i = 0; i < num; i++ )
		cout << "My name is " << name << endl;

	} catch ( ArgException& e)  // catch any exceptions
	{ cerr << "error: " << e.error() << " for arg " << e.argId() << endl; }
}

