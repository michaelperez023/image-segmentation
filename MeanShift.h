// MeanShift.h
//
// Mean shift filtering and segmentation of colour images in the joint
// spatial-colour domain, with uniform kernels: a disc of radius hs pixels in
// the image and a ball of radius hr in CIE L*a*b* colour space.
//
// Adapted from bbbbyang/Mean-Shift-Segmentation (MeanShift.h) by Bingyang
// Liu, https://github.com/bbbbyang/Mean-Shift-Segmentation. Modified by
// Michael Perez for CAP 4410 Computer Vision (Assignment 3, Florida
// Polytechnic University, fall 2019); bug fixes, the merging of small
// regions and tests added in 2026.
//
// Algorithm: D. Comaniciu and P. Meer, "Mean Shift: A Robust Approach Toward
// Feature Space Analysis", IEEE Transactions on Pattern Analysis and Machine
// Intelligence 24(5), 2002.

#ifndef MEANSHIFT_H
#define MEANSHIFT_H

#include <opencv2/core.hpp>

#include <vector>

// A point of the joint domain: a position in the image (x is the column, y
// the row) and an L*a*b* colour
class Point5D
{
public:
	double x = 0.0, y = 0.0;           // spatial value
	double l = 0.0, a = 0.0, b = 0.0;  // L*a*b* value

	Point5D() = default;
	Point5D(double px, double py, const cv::Vec3f& lab);

	void Accumulate(const Point5D& pt);                     // add another point
	void Scale(double scale);                               // multiply every value
	double SquaredColourDistance(const Point5D& pt) const;  // in L*a*b*
	double SquaredSpatialDistance(const Point5D& pt) const; // in pixels
	cv::Vec3f Colour() const;
};

class MeanShift
{
public:
	// The iterations for a pixel stop once the window's mean moves by at
	// most kSpatialTolerance pixels in the image and kColourTolerance in
	// L*a*b*, or after kMaxSteps steps.
	static constexpr double kSpatialTolerance = 0.3;
	static constexpr double kColourTolerance = 0.3;
	static constexpr int kMaxSteps = 100;

	// Result of MSFiltering
	struct Filtered
	{
		cv::Mat lab;              // CV_32FC3: colour of the mode each pixel converged to
		cv::Mat modes;            // CV_32FC2: (x, y) position of that mode
		double mean_steps = 0.0;  // mean shift steps per pixel, on average
		int max_steps = 0;        // the most steps any pixel took
		int unconverged = 0;      // pixels stopped by kMaxSteps (or by an empty window)
	};

	// Result of MSSegmentation
	struct Regions
	{
		cv::Mat labels;                  // CV_32S: region of each pixel, numbered from 0 in raster order
		std::vector<cv::Vec3f> colours;  // mean filtered colour of each region
		std::vector<int> sizes;          // pixels in each region
		int grown = 0;                   // regions grown before small ones were merged

		int Count() const { return static_cast<int>(sizes.size()); }
		cv::Mat Image() const;           // CV_32FC3: each pixel set to its region's colour
	};

	// hs: spatial bandwidth (radius) in pixels; hr: colour bandwidth in
	// L*a*b* units. Throws std::invalid_argument unless both are positive.
	MeanShift(double hs, double hr);

	double SpatialBandwidth() const { return hs; }
	double ColourBandwidth() const { return hr; }

	// Mean shift filtering of a CV_32FC3 L*a*b* image (see BgrToLab). For
	// each pixel a window starts at the pixel's position and colour. It
	// holds the pixels within hs of its centre in the image and within hr of
	// its colour; the centre moves to their mean position and colour, and
	// this repeats until the centre stops moving. The pixel takes the colour
	// of the point it converges to, its mode.
	Filtered MSFiltering(const cv::Mat& lab) const;

	// Segmentation of a filtered CV_32FC3 L*a*b* image. Regions are grown
	// from seed pixels taken in raster order: an 8-connected neighbour joins
	// a region when its colour is within hr of the seed's colour. Then, while
	// some region has fewer than min_region pixels, the smallest one is
	// merged into the neighbouring region with the closest mean colour
	// (min_region 0 or 1 merges nothing). Throws std::invalid_argument for a
	// negative min_region.
	Regions MSSegmentation(const cv::Mat& filtered, int min_region) const;

private:
	double hs;  // spatial radius
	double hr;  // colour radius
};

// Conversions between 8-bit BGR images, as read by cv::imread (BgrToLab also
// accepts 1- and 4-channel images), and CV_32FC3 L*a*b* images with L from 0
// to 100. LabToBgr rounds to the nearest 8-bit value.
cv::Mat BgrToLab(const cv::Mat& bgr);
cv::Mat LabToBgr(const cv::Mat& lab);

// Copy of an 8-bit BGR image with the region boundaries drawn in `colour`:
// a pixel is drawn when its right or lower neighbour is in another region.
cv::Mat DrawBoundaries(const cv::Mat& bgr, const cv::Mat& labels, const cv::Vec3b& colour);

#endif // MEANSHIFT_H
