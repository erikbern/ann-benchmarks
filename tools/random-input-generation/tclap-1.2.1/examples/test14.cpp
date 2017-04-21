#include "tclap/CmdLine.h"
#include <iterator>
#include <algorithm>

// Define a simple 3D vector type
template<typename T, size_t LEN>
struct Vect : public TCLAP::StringLikeTrait {
    //typedef TCLAP::StringLike ValueCategory;
    T v[LEN];

    // operator= will be used to assign to the vector
    Vect& operator=(const std::string &str)
        {
            std::istringstream iss(str);
            for (size_t n = 0; n < LEN; n++) {
                if (!(iss >> v[n])) {
                    std::ostringstream oss;
                    oss << " is not a vector of size " << LEN;
                    throw TCLAP::ArgParseException(str + oss.str());
                }
            }

            if (!iss.eof()) {
                std::ostringstream oss;
                oss << " is not a vector of size " << LEN;
                throw TCLAP::ArgParseException(str + oss.str());
            }

            return *this;
        }

    std::ostream& print(std::ostream &os) const
        {
            std::copy(v, v + LEN, std::ostream_iterator<T>(os, " "));
            return os;
        }

};

int main(int argc, char *argv[])
{
    TCLAP::CmdLine cmd("Command description message", ' ', "0.9");
    TCLAP::ValueArg< Vect<double, 3> > vec("v", "vect", "vector",
                                           true, Vect<double, 3>(),
                                           "3D vector", cmd);

    try {
	cmd.parse(argc, argv);
    } catch(std::exception &e) {
	std::cout << e.what() << std::endl;
	return EXIT_FAILURE;
    }

    vec.getValue().print(std::cout);
    std::cout << std::endl;
}
