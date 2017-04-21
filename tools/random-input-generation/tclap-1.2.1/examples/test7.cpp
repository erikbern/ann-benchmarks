#include <string>
#include "tclap/CmdLine.h"

using namespace TCLAP;
using namespace std;

int main(int argc, char **argv)
{
	// Wrap everything in a try block.  Do this every time,
	// because exceptions will be thrown for problems.
	try {

	// Define the command line object.
	CmdLine cmd("Command description message. This is a long multi-line message meant to test line wrapping.  This is more text that doesn't really do anything besides take up lots of space that otherwise might be used for something real.  That should be enough, don't you think?", ' ', "0.9");

	vector<string> allowed;
	allowed.push_back("homer");
	allowed.push_back("marge");
	allowed.push_back("bart");
	allowed.push_back("lisa");
	allowed.push_back("maggie");
	ValuesConstraint<string> vallowed( allowed );

	MultiArg<string> nameArg("n","name","Name to print. This is a long, nonsensical message to test line wrapping.  Hopefully it works.",true,&vallowed);
	cmd.add( nameArg );

	vector<int> iallowed;
	iallowed.push_back(1);
	iallowed.push_back(2);
	iallowed.push_back(3);
	ValuesConstraint<int> iiallowed( iallowed );

	UnlabeledMultiArg<int> intArg("times","Number of times to print",false,
					              &iiallowed);
	cmd.add( intArg );

	// Ignore the names and comments!  These  args mean nothing (to this
	// program) and are here solely to take up space.
    ValueArg<int> gapCreate("f","gapCreate", "The cost of creating a gap",
	                                 false, -10, "negative int");
	cmd.add( gapCreate );

	ValueArg<int> gapExtend("g","gap-Extend",
		"The cost for each extension of a gap", false, -2, "negative int");
	cmd.add( gapExtend );

	SwitchArg dna("d","isDna","The input sequences are DNA", false);
	cmd.add( dna );

	ValueArg<string> scoringMatrixName("s","scoring--Matrix",
		"Scoring Matrix name", false,"BLOSUM50","name string");
	cmd.add( scoringMatrixName );

	ValueArg<string> seq1Filename ("x","filename1",
		"Sequence 1 filename (FASTA format)", false,"","filename");
	cmd.add( seq1Filename );

	ValueArg<string> seq2Filename ("z","filename2",
		"Sequence 2 filename (FASTA format)", false,"","filename");
	cmd.add( seq2Filename );

	ValueArg<float> lowerBound("b","lowerBound", "lower percentage bound",
		false,1.0,"float lte 1");
	cmd.add( lowerBound );

	ValueArg<float> upperBound("u","upperBound", "upper percentage bound",
		false,1.0,"float lte 1");
	cmd.add( upperBound );

	ValueArg<int> limit("l","limit","Max number of alignments allowed",
		false, 1000,"int");
	cmd.add( limit );

	argv[0] = const_cast<char*>("ThisIsAVeryLongProgramNameDesignedToTestSpacePrintWhichUsedToHaveProblemsWithLongProgramNamesIThinkItIsNowLongEnough");

	// Parse the args.
	cmd.parse( argc, argv );

	// Get the value parsed by each arg.
	vector<int> num = intArg.getValue();

	for ( unsigned int i = 0; i < num.size(); i++ )
		cout << "Got num " << num[i] << endl;

	vector<string> name = nameArg.getValue();

	for ( unsigned int i = 0; i < name.size(); i++ )
		cout << "Got name " << name[i] << endl;


	} catch (ArgException& e)  // catch any exceptions
	{ cerr << "error: " << e.error() << " for arg " << e.argId() << endl; }
}

