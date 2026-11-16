// Otsu.h
//
// Otsu's method: grey-level thresholds that split an image's histogram into
// classes with the largest between-class variance. Works for two or more
// classes. With more than two classes every combination of thresholds is
// tried, and cumulative sums of the histogram make each one cost O(1).
//
// Michael Perez, CAP 4410 Computer Vision, Assignment 3, Florida Polytechnic
// University (fall 2019). Rewritten in 2026: the 2019 four-class search left
// the brightest class out of its score (see README.md).
//
// Algorithm: N. Otsu, "A Threshold Selection Method from Gray-Level
// Histograms", IEEE Transactions on Systems, Man, and Cybernetics 9(1), 1979.

#ifndef OTSU_H
#define OTSU_H

#include <opencv2/core.hpp>

#include <vector>

namespace otsu
{
	// Number of pixels at each grey level (256 levels for 8-bit images)
	using Histogram = std::vector<double>;

	// Thresholds t[0] < t[1] < ... < t[K-2] for K classes. Class k holds the
	// levels v with t[k-1] < v <= t[k], where t[-1] = -1 and t[K-1] is the
	// highest level, so a pixel equal to a threshold belongs to the lower
	// class, as with cv::threshold.
	struct Result
	{
		std::vector<int> thresholds;
		std::vector<double> class_sizes;  // pixels in each class
		double between_variance = 0.0;    // sigma_B^2 = sum_k w_k (mu_k - mu)^2
		double total_variance = 0.0;      // sigma_T^2 of the whole image

		// Separability eta = sigma_B^2 / sigma_T^2, from 0 to 1 (0 for a
		// constant image, where there is no variance to explain)
		double Eta() const;
	};

	// Histogram of an 8-bit, 1-channel image, computed afresh on each call.
	// Throws std::invalid_argument for an empty image or another type.
	Histogram ComputeHistogram(const cv::Mat& gray);

	// The thresholds for `classes` classes that maximise the between-class
	// variance. Every class covers at least one level (it may hold no
	// pixels). When several sets of thresholds give the same classes (the
	// levels between them are empty), the lowest is returned, so each
	// threshold is the highest occupied level of its class. Throws
	// std::invalid_argument unless 2 <= classes <= histogram size and the
	// histogram holds at least one pixel.
	//
	// The search tries all C(L - 1, classes - 1) sets of thresholds for L
	// levels: 255 for 2 classes and about 2.7 million for 4 classes of an
	// 8-bit image (a few milliseconds).
	Result FindThresholds(const Histogram& histogram, int classes);
	Result FindThresholds(const cv::Mat& gray, int classes);

	// sigma_B^2 of the classes given by `thresholds` (increasing, each from
	// 0 to histogram size - 2), computed directly from the class means
	double BetweenClassVariance(const Histogram& histogram, const std::vector<int>& thresholds);

	// Class index of every pixel of an 8-bit, 1-channel image (CV_8U, 0 to
	// K - 1), with the class boundaries described under Result
	cv::Mat ClassIndices(const cv::Mat& gray, const std::vector<int>& thresholds);

	// The same classes drawn as evenly spaced grey levels: class k of K
	// becomes 255 * k / (K - 1), so 2 classes give 0 and 255 and 4 classes
	// give 0, 85, 170 and 255.
	cv::Mat ClassImage(const cv::Mat& gray, const std::vector<int>& thresholds);

	// 8-bit grayscale copy of an 8-bit 1-, 3- (BGR) or 4-channel (BGRA) image
	cv::Mat ToGray(const cv::Mat& image);

	// True for a 1-channel image, or one whose B, G and R values are equal
	// at every pixel
	bool IsGray(const cv::Mat& image);

	// Two-class Otsu applied to each channel of an 8-bit image separately
	// (the 2019 program's method for colour images). Each channel becomes 0
	// or 255, so a colour image can have up to 8 colours. The result for
	// each channel is added to `results` when it is not null.
	cv::Mat ThresholdChannels(const cv::Mat& image, std::vector<Result>* results = nullptr);
}

#endif // OTSU_H
