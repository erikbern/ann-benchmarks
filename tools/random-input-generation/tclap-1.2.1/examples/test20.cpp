#include <string>
#include <iostream>
#include <algorithm>
#include <tclap/CmdLine.h>

using namespace TCLAP;
using namespace std;

int main(int argc, char** argv)
{
    // Wrap everything in a try block.  Do this every time,
    // because exceptions will be thrown for problems.
    try { 

    // Define the command line object.
    CmdLine cmd("Command description message", '=', "0.9");

    SwitchArg atmcSwitch("a", "atmc", "aContinuous time semantics", false);
    SwitchArg btmcSwitch("b", "btmc", "bDiscrete time semantics",   false);
    cmd.xorAdd(atmcSwitch, btmcSwitch);

    // Parse the args.
    cmd.parse( argc, argv );

    } catch (ArgException &e)  // catch any exceptions
    { cerr << "error: " << e.error() << " for arg " << e.argId() << endl; }
}
