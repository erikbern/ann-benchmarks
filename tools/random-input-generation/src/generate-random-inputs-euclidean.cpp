/**
 * @file generate-random-inputs-euclidean.cpp
 * @author Martin Aumüller
 *
 */

#include <algorithm>
#include <cmath>
#include <random>
#include <utility>
#include <fstream>
#include <iostream>
#include <set>
#include <tclap/CmdLine.h>

std::random_device rd;
std::mt19937_64 gen(rd());
std::normal_distribution<double> distribution(0.0, 1.0);

using vecType = std::vector<double>;

double length(const vecType& v) {
    double t = 0.0;
    for (auto e: v) {
        t += e * e;
    }
    return sqrt(t);
}

void normalize(vecType& v) {
    auto len = length(v);
    for (size_t i = 0; i < v.size(); i++) {
        v[i] /= len;
    }
}

vecType generateRandomVector(size_t d, bool unit) {
    vecType v(d);
    for (size_t i = 0; i < d; i++)
        v[i] = distribution(gen) / sqrt(d);
    if (unit)
        normalize(v);
    return v;
}

vecType distortPoint(const vecType& v, float targetDistance, bool unit) {
    size_t d = v.size();
    vecType w(d);
    for (size_t i = 0; i < d; i++)
        w[i] = v[i] + distribution(gen) * targetDistance/sqrt(d);
    if (unit)
        normalize(w);
    return w;
}

template <typename T>
void writeFile(std::string filename, std::set<T>& set) {
	std::ofstream outputFile;
	outputFile.open(filename);
    for (auto& q: set) {
        for (auto& e: q) {
            outputFile  << e << " ";
        }
        outputFile << std::endl;
    }
	outputFile.close();
}

int main(int argc, char** argv) {
	TCLAP::CmdLine cmd("This program generates random inputs in Euclidean space.", ' ', "0.9");

	TCLAP::ValueArg<size_t> numberOfPointsArg("n", "numpoints", "Total number of points", true, 10, "integer");
	TCLAP::ValueArg<size_t> numberOfDimensionsArg("d", "dimension", "Number of dimensions", true, 15, "integer");
	TCLAP::ValueArg<size_t> numberOfClusters("c", "numclusters", "Number of clusters", true, 15, "integer");
	TCLAP::ValueArg<size_t> pointsPerCluster("p", "pointscluster", "Number of points per cluster", true, 15, "integer");
	TCLAP::ValueArg<std::string> outputFileArg("o", "outputfile", "File the output should be written to. ", true, "input.txt", "string");
	TCLAP::ValueArg<std::string> queryOutFileArg("q", "queryoutfile", "File the query set should be written to. ", true, "input.txt", "string");
    TCLAP::ValueArg<size_t> seedArg("s", "seed", "Seed for random generator", false, 1234, "integer");
    TCLAP::SwitchArg randomQueryArg("g", "randomqueries", "Should random query points (without close neighbors) be added?");
    TCLAP::SwitchArg normalizeArg("N", "normalize", "Normalize vectors to unit length?");

    cmd.add(numberOfPointsArg);
    cmd.add(numberOfDimensionsArg);
    cmd.add(numberOfClusters);
    cmd.add(pointsPerCluster);
    cmd.add(outputFileArg);
    cmd.add(queryOutFileArg);
    cmd.add(seedArg);
    cmd.add(randomQueryArg);
    cmd.add(normalizeArg);

	cmd.parse(argc, argv);

	auto d = numberOfDimensionsArg.getValue();
	auto n = numberOfPointsArg.getValue();
    auto numClusters = numberOfClusters.getValue();
    auto numPoints = pointsPerCluster.getValue();
    size_t numNoisePoints = 2 * numPoints;
    bool unit = normalizeArg.isSet();

    if (seedArg.isSet()) {
       gen.seed(seedArg.getValue());
    }

	std::set<vecType> set;
    std::set<vecType> querySet;

    for (size_t i = 0; i < numClusters; i++) {
        querySet.insert(generateRandomVector(d, unit));
    }

    std::cout << "Generated " << querySet.size() << " clusters ... " << std::endl;

    double windowSize = sqrt(2) / (3.0 * numClusters);

    int curQuery = 0;

    for (auto& q: querySet) {
        curQuery++;
        double targetDistance = curQuery * windowSize;

        // Compute points at target distance around query.
        for (size_t i = 0; i < numPoints; i++) {
            set.insert(distortPoint(q, targetDistance, unit));
        }
        // Compute some noise.
        targetDistance *= 2;
        for (size_t i = 0; i < numNoisePoints; i++) {
            set.insert(distortPoint(q, targetDistance, unit));
        }
    }

    // Add random query points
    if (randomQueryArg.isSet()) {
        for (size_t i = 0; i < numClusters; i++) {
            querySet.insert(generateRandomVector(d, unit));
        }
    }

    std::cout << "Created clusters with a total number of " << set.size() << " points." << std::endl;
	std::cout << "Fill with random points..." << std::endl;

	// Fill with random points
    while (set.size() < n) {
        set.insert(generateRandomVector(d, unit));
	}

    std::cout << "Writing result to file" << std::endl;
    writeFile(outputFileArg.getValue(), set);
    writeFile(queryOutFileArg.getValue(), querySet);

	return 0;

}
