#include "tclap/CmdLine.h"
#include <iterator>
#include <algorithm>

using namespace TCLAP;

// Define a simple 3D vector type
struct Vect3D {
    double v[3];

    // operator= will be used to assign to the vector
    Vect3D& operator=(const std::string &str)
    {
	std::istringstream iss(str);
	if (!(iss >> v[0] >> v[1] >> v[2]))
	    throw TCLAP::ArgParseException(str + " is not a 3D vector");

	return *this;
    }

    std::ostream& print(std::ostream &os) const
    {
	std::copy(v, v + 3, std::ostream_iterator<double>(os, " "));
	return os;
    }

};

std::ostream& operator<<(std::ostream &os, const Vect3D &v)
{
    return v.print(os);
}

// Create an ArgTraits for the 3D vector type that declares it to be
// of string like type
namespace TCLAP {
template<>
struct ArgTraits<Vect3D> {
    typedef StringLike ValueCategory;
};
}

int main(int argc, char *argv[])
{
    CmdLine cmd("Command description message", ' ', "0.9");
    MultiArg<Vect3D> vec("v", "vect", "vector", 
			 true, "3D vector", cmd);
    
    try {
	cmd.parse(argc, argv);
    } catch(std::exception &e) {
	std::cout << e.what() << std::endl;
	return EXIT_FAILURE;
    }

    std::copy(vec.begin(), vec.end(),
	      std::ostream_iterator<Vect3D>(std::cout, "\n"));

    std::cout << "REVERSED" << std::endl;

    // use alt. form getValue()
    std::vector<Vect3D> v(vec.getValue());
    std::reverse(v.begin(), v.end());

    std::copy(v.begin(), v.end(),
	      std::ostream_iterator<Vect3D>(std::cout, "\n"));
}

