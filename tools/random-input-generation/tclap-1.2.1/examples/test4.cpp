

#include "tclap/CmdLine.h"
#include "tclap/DocBookOutput.h"
#include "tclap/ZshCompletionOutput.h"
#include <iostream>
#include <string>

using namespace TCLAP;
using namespace std;


// This exemplifies how the output class can be overridden to provide
// user defined output.
class MyOutput : public StdOutput
{
	public:

		virtual void failure(CmdLineInterface& c, ArgException& e)
		{
			static_cast<void>(c); // Ignore input, don't warn
			cerr << "my failure message: " << endl
			     << e.what() << endl;
			exit(1);
		}

		virtual void usage(CmdLineInterface& c)
		{
			cout << "my usage message:" << endl;
			list<Arg*> args = c.getArgList();
			for (ArgListIterator it = args.begin(); it != args.end(); it++)
				cout << (*it)->longID()
					 << "  (" << (*it)->getDescription() << ")" << endl;
		}

		virtual void version(CmdLineInterface& c)
		{
			static_cast<void>(c); // Ignore input, don't warn
			cout << "my version message: 0.1" << endl;
		}
};


bool _boolTestB;
bool _boolTestA;
string _stringTest;

void parseOptions(int argc, char** argv);

int main(int argc, char** argv)
{

	parseOptions(argc,argv);

	cout << "for string we got : " << _stringTest<< endl
		 << "for bool B we got : " << _boolTestB << endl
		 << "for bool A we got : " << _boolTestA << endl;

}


void parseOptions(int argc, char** argv)
{
	try {

	CmdLine cmd("this is a message", ' ', "0.99" );

	// set the output
	MyOutput my;
	//ZshCompletionOutput my;
	//DocBookOutput my;
	cmd.setOutput(&my);

	//
	// Define arguments
	//

	SwitchArg btest("B","sB", "exist Test B", false);
	SwitchArg atest("A","sA", "exist Test A", false);

	ValueArg<string> stest("s", "Bs", "string test", true, "homer",
					       "string");
	cmd.add( stest );
	cmd.add( btest );
	cmd.add( atest );

	//
	// Parse the command line.
	//
	cmd.parse(argc,argv);


	//
	// Set variables
	//
	_stringTest = stest.getValue();
	_boolTestB = btest.getValue();
	_boolTestA = atest.getValue();


	} catch ( ArgException& e )
	{ cout << "ERROR: " << e.error() << " " << e.argId() << endl; }
}



