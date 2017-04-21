#include <iostream>
#include <string>

#include <tclap/CmdLine.h>

using namespace TCLAP;

//
// This file tests that we can parse args from a vector
// of strings rather than argv.  This also tests a bug
// where a single element in the vector contains both
// the flag and value AND the value contains the flag 
// from another switch arg.  This would fool the parser
// into thinking that the string was a combined switches
// string rather than a flag value combo.
//
// This should not print an error
//
// Contributed by Nico Lugil.
//
int main()
{

   try
   {
      CmdLine cmd("Test", ' ', "not versioned",true);

      MultiArg<std::string> Arg("X","fli","fli module",false,"string");
      cmd.add(Arg);
      MultiSwitchArg ArgMultiSwitch("d","long_d","example");
      cmd.add(ArgMultiSwitch);

      std::vector<std::string> in;
      in.push_back("prog name");
      in.push_back("-X module");
      cmd.parse(in);

      std::vector<std::string> s = Arg.getValue();
      for(unsigned int i = 0 ; i < s.size() ; i++)
      {
         std::cout << s[i] << "\n";
      }
      std::cout << "MultiSwtichArg was found " << ArgMultiSwitch.getValue() << " times.\n";

   }
   catch (ArgException &e)  // catch any exceptions
   {
      std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
   }

   std::cout << "done...\n";

   return 0;
}



