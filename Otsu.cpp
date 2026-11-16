// Otsu.cpp
//
// Otsu's method for two or more classes; see Otsu.h.
//
// Michael Perez, CAP 4410 Computer Vision, Assignment 3, Florida Polytechnic
// University (fall 2019). Rewritten in 2026.

#include "Otsu.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace otsu
{
	namespace
	{
		constexpr int kLevels8U = 256;

		// Cumulative sums of a histogram: count[i] is the number of pixels
		// with a level below i, and sum[i] the sum of their levels.
		//
		// For N pixels with mean level mu, and N_k pixels with level sum S_k
		// in class k,
		//   sigma_B^2 = sum_k (N_k / N) (S_k / N_k - mu)^2
		//             = (sum_k S_k^2 / N_k) / N - mu^2,
		// so the thresholds that maximise sum_k S_k^2 / N_k also maximise
		// sigma_B^2, and each term needs only two differences of the sums.
		class CumulativeSums
		{
		public:
			explicit CumulativeSums(const Histogram& histogram)
				: count(histogram.size() + 1, 0.0), sum(histogram.size() + 1, 0.0)
			{
				for (std::size_t i = 0; i < histogram.size(); ++i)
				{
					count[i + 1] = count[i] + histogram[i];
					sum[i + 1] = sum[i] + static_cast<double>(i) * histogram[i];
				}
			}

			// S^2 / N for the class of levels first to last; 0 if it has no pixels
			double Term(int first, int last) const
			{
				const double n = count[last + 1] - count[first];
				if (n <= 0.0)
					return 0.0;
				const double s = sum[last + 1] - sum[first];
				return s * s / n;
			}

		private:
			std::vector<double> count, sum;
		};

		// Tries every set of thresholds, choosing them from the lowest class up
		class ThresholdSearch
		{
		public:
			ThresholdSearch(const Histogram& histogram, int num_classes)
				: sums(histogram), levels(static_cast<int>(histogram.size())), classes(num_classes),
				  current(num_classes - 1), last_class_term(histogram.size() + 1, 0.0)
			{
				// The term of the last class depends only on its first level
				for (int first = 0; first < levels; ++first)
					last_class_term[first] = sums.Term(first, levels - 1);
			}

			std::vector<int> Run()
			{
				Choose(0, 0, 0.0);
				return best;
			}

		private:
			// Chooses t[k], the highest level of class k, which starts at level
			// `first`; `score` is the sum of the terms of classes 0 to k - 1.
			// Only a strictly better score replaces the best so far, so among
			// equal scores the first (lowest) thresholds are kept.
			void Choose(int k, int first, double score)
			{
				// Leave at least one level for each later class
				const int highest = levels - classes + k;
				if (k == classes - 2)
				{
					// The last threshold also fixes the last class
					for (int t = first; t <= highest; ++t)
					{
						const double total = score + sums.Term(first, t) + last_class_term[t + 1];
						if (total > best_score)
						{
							best_score = total;
							current[k] = t;
							best = current;
						}
					}
					return;
				}
				for (int t = first; t <= highest; ++t)
				{
					current[k] = t;
					Choose(k + 1, t + 1, score + sums.Term(first, t));
				}
			}

			CumulativeSums sums;
			int levels;
			int classes;
			std::vector<int> current, best;
			std::vector<double> last_class_term;
			double best_score = -1.0;
		};

		void CheckHistogram(const Histogram& histogram)
		{
			double total = 0.0;
			for (double count : histogram)
			{
				if (!std::isfinite(count) || count < 0.0)
					throw std::invalid_argument("otsu: histogram counts must be finite and not negative");
				total += count;
			}
			if (total <= 0.0)
				throw std::invalid_argument("otsu: the histogram holds no pixels");
		}

		void CheckThresholds(const std::vector<int>& thresholds, int levels)
		{
			if (thresholds.empty())
				throw std::invalid_argument("otsu: at least one threshold is needed");
			int previous = -1;
			for (int t : thresholds)
			{
				if (t <= previous || t > levels - 2)
					throw std::invalid_argument("otsu: thresholds must increase and lie from 0 to " +
					                            std::to_string(levels - 2));
				previous = t;
			}
		}

		void CheckGray(const cv::Mat& gray)
		{
			if (gray.empty() || gray.type() != CV_8UC1)
				throw std::invalid_argument("otsu: expected a non-empty 8-bit, 1-channel image");
		}

		// Number of pixels and mean level
		void Moments(const Histogram& histogram, double& total, double& mean)
		{
			total = 0.0;
			double sum = 0.0;
			for (std::size_t i = 0; i < histogram.size(); ++i)
			{
				total += histogram[i];
				sum += static_cast<double>(i) * histogram[i];
			}
			mean = sum / total;
		}

		double TotalVariance(const Histogram& histogram)
		{
			double total = 0.0, mean = 0.0;
			Moments(histogram, total, mean);
			double variance = 0.0;
			for (std::size_t i = 0; i < histogram.size(); ++i)
			{
				const double d = static_cast<double>(i) - mean;
				variance += histogram[i] * d * d;
			}
			return variance / total;
		}

		std::vector<double> ClassSizes(const Histogram& histogram, const std::vector<int>& thresholds)
		{
			std::vector<double> sizes(thresholds.size() + 1, 0.0);
			std::size_t k = 0;
			for (std::size_t i = 0; i < histogram.size(); ++i)
			{
				while (k < thresholds.size() && static_cast<int>(i) > thresholds[k])
					++k;
				sizes[k] += histogram[i];
			}
			return sizes;
		}

		// Lookup table from grey level to class: level v goes to the number
		// of thresholds below it, then through `class_value`
		template <typename ClassValue>
		cv::Mat ClassLookup(const std::vector<int>& thresholds, ClassValue class_value)
		{
			CheckThresholds(thresholds, kLevels8U);
			cv::Mat lut(1, kLevels8U, CV_8U);
			std::size_t k = 0;
			for (int v = 0; v < kLevels8U; ++v)
			{
				while (k < thresholds.size() && v > thresholds[k])
					++k;
				lut.at<uchar>(v) = class_value(static_cast<int>(k));
			}
			return lut;
		}
	}

	double Result::Eta() const
	{
		if (total_variance <= 0.0)
			return 0.0;
		return std::min(1.0, between_variance / total_variance);
	}

	Histogram ComputeHistogram(const cv::Mat& gray)
	{
		CheckGray(gray);
		std::vector<std::int64_t> counts(kLevels8U, 0);
		for (int r = 0; r < gray.rows; ++r)
		{
			const uchar* row = gray.ptr<uchar>(r);
			for (int c = 0; c < gray.cols; ++c)
				++counts[row[c]];
		}
		return Histogram(counts.begin(), counts.end());
	}

	Result FindThresholds(const Histogram& histogram, int classes)
	{
		const int levels = static_cast<int>(histogram.size());
		if (classes < 2 || classes > levels)
			throw std::invalid_argument("otsu: the number of classes must be from 2 to the number of levels (" +
			                            std::to_string(levels) + ")");
		CheckHistogram(histogram);

		Result result;
		result.thresholds = ThresholdSearch(histogram, classes).Run();
		result.class_sizes = ClassSizes(histogram, result.thresholds);
		result.between_variance = BetweenClassVariance(histogram, result.thresholds);
		result.total_variance = TotalVariance(histogram);
		return result;
	}

	Result FindThresholds(const cv::Mat& gray, int classes)
	{
		return FindThresholds(ComputeHistogram(gray), classes);
	}

	double BetweenClassVariance(const Histogram& histogram, const std::vector<int>& thresholds)
	{
		const int levels = static_cast<int>(histogram.size());
		CheckThresholds(thresholds, levels);
		CheckHistogram(histogram);
		double total = 0.0, mean = 0.0;
		Moments(histogram, total, mean);

		double variance = 0.0;
		int first = 0;
		for (std::size_t k = 0; k <= thresholds.size(); ++k)
		{
			const int last = k < thresholds.size() ? thresholds[k] : levels - 1;
			double n = 0.0, sum = 0.0;
			for (int i = first; i <= last; ++i)
			{
				n += histogram[i];
				sum += i * histogram[i];
			}
			if (n > 0.0)
			{
				const double d = sum / n - mean;
				variance += n * d * d;
			}
			first = last + 1;
		}
		return variance / total;
	}

	cv::Mat ClassIndices(const cv::Mat& gray, const std::vector<int>& thresholds)
	{
		CheckGray(gray);
		const cv::Mat lut = ClassLookup(thresholds, [](int k) { return static_cast<uchar>(k); });
		cv::Mat indices;
		cv::LUT(gray, lut, indices);
		return indices;
	}

	cv::Mat ClassImage(const cv::Mat& gray, const std::vector<int>& thresholds)
	{
		CheckGray(gray);
		// ClassLookup checks that there is at least one threshold
		const double classes = static_cast<double>(thresholds.size()) + 1.0;
		const cv::Mat lut = ClassLookup(thresholds,
		                                [classes](int k) { return cv::saturate_cast<uchar>(255.0 * k / (classes - 1.0)); });
		cv::Mat image;
		cv::LUT(gray, lut, image);
		return image;
	}

	cv::Mat ToGray(const cv::Mat& image)
	{
		if (image.empty() || image.depth() != CV_8U)
			throw std::invalid_argument("otsu: expected a non-empty 8-bit image");
		cv::Mat gray;
		switch (image.channels())
		{
		case 1:
			gray = image.clone();
			break;
		case 3:
			cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
			break;
		case 4:
			cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
			break;
		default:
			throw std::invalid_argument("otsu: expected a 1-, 3- or 4-channel image");
		}
		return gray;
	}

	bool IsGray(const cv::Mat& image)
	{
		if (image.channels() == 1)
			return true;
		if (image.channels() != 3 && image.channels() != 4)
			throw std::invalid_argument("otsu: expected a 1-, 3- or 4-channel image");
		std::vector<cv::Mat> channels;
		cv::split(image, channels);
		return cv::countNonZero(channels[0] != channels[1]) == 0 && cv::countNonZero(channels[1] != channels[2]) == 0;
	}

	cv::Mat ThresholdChannels(const cv::Mat& image, std::vector<Result>* results)
	{
		if (image.empty() || image.depth() != CV_8U)
			throw std::invalid_argument("otsu: expected a non-empty 8-bit image");
		cv::Mat source = image;
		if (image.channels() == 4)
			cv::cvtColor(image, source, cv::COLOR_BGRA2BGR);

		std::vector<cv::Mat> channels;
		cv::split(source, channels);
		for (cv::Mat& channel : channels)
		{
			const Result result = FindThresholds(channel, 2);
			channel = ClassImage(channel, result.thresholds);
			if (results)
				results->push_back(result);
		}
		cv::Mat merged;
		cv::merge(channels, merged);
		return merged;
	}
}
