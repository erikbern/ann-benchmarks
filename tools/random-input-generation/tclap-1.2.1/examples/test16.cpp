#include "tclap/CmdLine.h"
#include <iterator>
#include <algorithm>

namespace TCLAP {
    template<>
    struct ArgTraits< std::vector<double> > {
        typedef StringLike ValueCategory;
    };

    template<>
    void SetString< std::vector<double> >(std::vector<double> &v,
                                          const std::string &s)
    {
        std::istringstream iss(s);
        while (iss) {
            double tmp;
            iss >> tmp;
            v.push_back(tmp);
        }
    }
}

int main(int argc, char *argv[])
{
    TCLAP::CmdLine cmd("Command description message", ' ', "0.9");
    TCLAP::ValueArg< std::vector<double> > vec("v", "vect", "vector",
                                               true,  std::vector<double>(),
                                               "3D vector", cmd);
    try {
	cmd.parse(argc, argv);
    } catch(std::exception &e) {
	std::cout << e.what() << std::endl;
	return EXIT_FAILURE;
    }

    const std::vector<double> &v = vec.getValue();
    std::copy(v.begin(), v.end(),
              std::ostream_iterator<double>(std::cout, "\n"));
    std::cout << std::endl;
}
