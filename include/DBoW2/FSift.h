#ifndef __D_T_F_SIFT_
#define __D_T_F_SIFT

// #define SIFT_64 1
#include <opencv2/core.hpp>
#include <vector>
#include <string>

#include "FClass.h"

namespace DBoW2
{
class FSift: protected FClass
{
public:
    // Descriptor type
    typedef std::vector<float > TDescriptor;
    // Pointer to a single descriptor
    typedef const TDescriptor *pDescriptor;
    // Descritor length
    #ifdef SIFT_64
    static const int L = 64;
    #else
    static const int L = 128;
    #endif

    // return the number of dimensions of the descriptor space
    inline static int demensions()
    {
        return L;
    }

    /**
     * Calculates the mean value of a set of descriptors
     * @param descriptors vector of pointers to descriptors
     * @param mean mean descriptor
    */
    static void meanValue(const std::vector<pDescriptor> &descriptors, 
        TDescriptor &mean);
    /**
     * Calculates the (squared) distance between two descriptors
     * @param a
     * @param b
     * @return (squared) distance
    */
    static double distance(const TDescriptor &a, const TDescriptor &b);

    /**
     * Returns a string version of the descriptor
     * @param a descriptor
     * @return string version
     */
    static std::string toString(const TDescriptor &a);

    /**
     * Returns a descriptor from a string
     * @param a descriptor
     * @param s string version
    */
    static void fromString(TDescriptor &a, const std::string &s);

    /**
     * Returns a mat with the descriptors in float format
     * @param descriptors
     * @param mat (out) NxL 32F matrix
    */
    static void toMat32F(const std::vector<TDescriptor> &descriptors, 
        cv::Mat &mat);
};
}
#endif