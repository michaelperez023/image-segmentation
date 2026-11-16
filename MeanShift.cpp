// MeanShift.cpp
//
// Mean shift filtering and segmentation; see MeanShift.h.
//
// Adapted from bbbbyang/Mean-Shift-Segmentation (MeanShift.cpp) by Bingyang
// Liu, https://github.com/bbbbyang/Mean-Shift-Segmentation. Modified by
// Michael Perez for CAP 4410 Computer Vision (Assignment 3, Florida
// Polytechnic University, fall 2019); bug fixes, the merging of small
// regions and tests added in 2026.

#include "MeanShift.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{
	// Row and column offsets of the 8 neighbours, for region growing
	const int dxdy[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

	void CheckLabImage(const cv::Mat& lab, const char* function)
	{
		if (lab.empty() || lab.type() != CV_32FC3)
			throw std::invalid_argument(std::string(function) + ": expected a non-empty CV_32FC3 L*a*b* image");
	}

	Point5D PointAt(const cv::Mat& lab, int row, int col)
	{
		return Point5D(col, row, lab.at<cv::Vec3f>(row, col));
	}

	// Merges every region with fewer than min_region pixels into a
	// neighbour. The smallest such region (the lowest label among equals)
	// goes first, into the 8-connected neighbour with the closest mean
	// colour; the merged region is then a candidate again if it is still
	// too small. Finally the regions are renumbered from 0 in raster order.
	// Returns the number of regions.
	int MergeSmallRegions(const cv::Mat& filtered, cv::Mat& labels, int count, int min_region)
	{
		struct Region
		{
			int size = 0;
			cv::Vec3d sum;             // sum of the filtered colours
			std::set<int> neighbours;  // labels of the adjacent regions
		};
		std::vector<Region> regions(count);
		const int rows = labels.rows, cols = labels.cols;
		for (int i = 0; i < rows; ++i)
		{
			for (int j = 0; j < cols; ++j)
			{
				const int label = labels.at<int>(i, j);
				regions[label].size += 1;
				regions[label].sum += cv::Vec3d(filtered.at<cv::Vec3f>(i, j));

				// Right, lower left, lower and lower right: each pair of
				// 8-connected pixels is seen once
				const int forward[4][2] = {{0, 1}, {1, -1}, {1, 0}, {1, 1}};
				for (const auto& d : forward)
				{
					const int r = i + d[0], c = j + d[1];
					if (r >= rows || c < 0 || c >= cols)
						continue;
					const int other = labels.at<int>(r, c);
					if (other != label)
					{
						regions[label].neighbours.insert(other);
						regions[other].neighbours.insert(label);
					}
				}
			}
		}

		std::vector<int> merged_into(count);
		for (int r = 0; r < count; ++r)
			merged_into[r] = r;

		// Regions that are too small, ordered by (size, label)
		std::set<std::pair<int, int>> small;
		for (int r = 0; r < count; ++r)
			if (regions[r].size < min_region)
				small.insert({regions[r].size, r});

		while (!small.empty())
		{
			const int r = small.begin()->second;
			small.erase(small.begin());
			Region& region = regions[r];
			if (region.neighbours.empty())
				continue;  // the whole image is one region

			// Closest mean colour; the set is in label order, so the lowest
			// label wins a tie
			const cv::Vec3d colour = region.sum / region.size;
			int target = -1;
			double best = std::numeric_limits<double>::infinity();
			for (int n : region.neighbours)
			{
				const cv::Vec3d difference = regions[n].sum / regions[n].size - colour;
				const double distance = difference.dot(difference);
				if (distance < best)
				{
					best = distance;
					target = n;
				}
			}

			Region& into = regions[target];
			if (into.size < min_region)
				small.erase({into.size, target});
			into.size += region.size;
			into.sum += region.sum;
			for (int n : region.neighbours)
			{
				regions[n].neighbours.erase(r);
				if (n != target)
				{
					regions[n].neighbours.insert(target);
					into.neighbours.insert(n);
				}
			}
			region = Region();
			merged_into[r] = target;
			if (into.size < min_region)
				small.insert({into.size, target});
		}

		// Renumber the remaining regions in raster order
		std::vector<int> new_label(count, -1);
		int next = 0;
		for (int i = 0; i < rows; ++i)
		{
			for (int j = 0; j < cols; ++j)
			{
				int& label = labels.at<int>(i, j);
				int r = label;
				while (merged_into[r] != r)
					r = merged_into[r];
				if (new_label[r] < 0)
					new_label[r] = next++;
				label = new_label[r];
			}
		}
		return next;
	}
}

Point5D::Point5D(double px, double py, const cv::Vec3f& lab) : x(px), y(py), l(lab[0]), a(lab[1]), b(lab[2])
{
}

void Point5D::Accumulate(const Point5D& pt)
{
	x += pt.x;
	y += pt.y;
	l += pt.l;
	a += pt.a;
	b += pt.b;
}

void Point5D::Scale(double scale)
{
	x *= scale;
	y *= scale;
	l *= scale;
	a *= scale;
	b *= scale;
}

double Point5D::SquaredColourDistance(const Point5D& pt) const
{
	return (l - pt.l) * (l - pt.l) + (a - pt.a) * (a - pt.a) + (b - pt.b) * (b - pt.b);
}

double Point5D::SquaredSpatialDistance(const Point5D& pt) const
{
	return (x - pt.x) * (x - pt.x) + (y - pt.y) * (y - pt.y);
}

cv::Vec3f Point5D::Colour() const
{
	return cv::Vec3f(static_cast<float>(l), static_cast<float>(a), static_cast<float>(b));
}

MeanShift::MeanShift(double spatial, double colour) : hs(spatial), hr(colour)
{
	if (!(hs > 0.0) || !(hr > 0.0) || !std::isfinite(hs) || !std::isfinite(hr))
		throw std::invalid_argument("MeanShift: the bandwidths must be positive");
}

MeanShift::Filtered MeanShift::MSFiltering(const cv::Mat& lab) const
{
	CheckLabImage(lab, "MeanShift::MSFiltering");
	const int rows = lab.rows, cols = lab.cols;
	const double hs2 = hs * hs, hr2 = hr * hr;
	const double spatial_tolerance2 = kSpatialTolerance * kSpatialTolerance;
	const double colour_tolerance2 = kColourTolerance * kColourTolerance;

	Filtered result;
	result.lab.create(lab.size(), CV_32FC3);
	result.modes.create(lab.size(), CV_32FC2);
	long long total_steps = 0;

	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			Point5D PtCur = PointAt(lab, i, j);  // the window's centre, starting at the pixel
			int step = 0;
			bool converged = false;
			while (!converged && step < kMaxSteps)
			{
				// Pixels within hs of the centre (a disc around it, wherever
				// it has moved) and within hr of its colour
				const int top = std::max(0, static_cast<int>(std::ceil(PtCur.y - hs)));
				const int bottom = std::min(rows - 1, static_cast<int>(std::floor(PtCur.y + hs)));
				const int left = std::max(0, static_cast<int>(std::ceil(PtCur.x - hs)));
				const int right = std::min(cols - 1, static_cast<int>(std::floor(PtCur.x + hs)));
				Point5D PtSum;
				int NumPts = 0;
				for (int hy = top; hy <= bottom; ++hy)
				{
					const cv::Vec3f* row = lab.ptr<cv::Vec3f>(hy);
					const double dy2 = (hy - PtCur.y) * (hy - PtCur.y);
					for (int hx = left; hx <= right; ++hx)
					{
						if ((hx - PtCur.x) * (hx - PtCur.x) + dy2 > hs2)
							continue;
						const Point5D Pt(hx, hy, row[hx]);
						if (Pt.SquaredColourDistance(PtCur) < hr2)
						{
							PtSum.Accumulate(Pt);
							++NumPts;
						}
					}
				}
				// The first window always holds the pixel itself; a later one
				// could in principle be empty
				if (NumPts == 0)
					break;

				PtSum.Scale(1.0 / NumPts);
				++step;
				// Both the position and the colour must have settled
				converged = PtSum.SquaredSpatialDistance(PtCur) <= spatial_tolerance2 &&
				            PtSum.SquaredColourDistance(PtCur) <= colour_tolerance2;
				PtCur = PtSum;
			}

			total_steps += step;
			result.max_steps = std::max(result.max_steps, step);
			if (!converged)
				++result.unconverged;
			result.lab.at<cv::Vec3f>(i, j) = PtCur.Colour();
			result.modes.at<cv::Vec2f>(i, j) = cv::Vec2f(static_cast<float>(PtCur.x), static_cast<float>(PtCur.y));
		}
	}
	result.mean_steps = static_cast<double>(total_steps) / (static_cast<double>(rows) * cols);
	return result;
}

MeanShift::Regions MeanShift::MSSegmentation(const cv::Mat& filtered, int min_region) const
{
	CheckLabImage(filtered, "MeanShift::MSSegmentation");
	if (min_region < 0)
		throw std::invalid_argument("MeanShift::MSSegmentation: min_region must not be negative");
	const int rows = filtered.rows, cols = filtered.cols;
	const double hr2 = hr * hr;

	// Region growing from seeds in raster order
	cv::Mat labels(filtered.size(), CV_32S, cv::Scalar(-1));
	int label = -1;
	std::vector<cv::Point> NeighbourPoints;
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			if (labels.at<int>(i, j) >= 0)
				continue;
			labels.at<int>(i, j) = ++label;
			const Point5D seed = PointAt(filtered, i, j);
			NeighbourPoints.assign(1, cv::Point(j, i));
			while (!NeighbourPoints.empty())
			{
				const cv::Point pt = NeighbourPoints.back();
				NeighbourPoints.pop_back();
				for (const auto& d : dxdy)
				{
					const int hy = pt.y + d[0], hx = pt.x + d[1];
					if (hy < 0 || hx < 0 || hy >= rows || hx >= cols || labels.at<int>(hy, hx) >= 0)
						continue;
					if (PointAt(filtered, hy, hx).SquaredColourDistance(seed) < hr2)
					{
						labels.at<int>(hy, hx) = label;
						NeighbourPoints.push_back(cv::Point(hx, hy));
					}
				}
			}
		}
	}

	Regions regions;
	regions.grown = label + 1;
	const int count = MergeSmallRegions(filtered, labels, regions.grown, min_region);

	// Size and mean colour of each region
	regions.sizes.assign(count, 0);
	std::vector<cv::Vec3d> sums(count);
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			const int r = labels.at<int>(i, j);
			regions.sizes[r] += 1;
			sums[r] += cv::Vec3d(filtered.at<cv::Vec3f>(i, j));
		}
	}
	regions.colours.resize(count);
	for (int r = 0; r < count; ++r)
		regions.colours[r] = cv::Vec3f(sums[r] / regions.sizes[r]);
	regions.labels = labels;
	return regions;
}

cv::Mat MeanShift::Regions::Image() const
{
	cv::Mat image(labels.size(), CV_32FC3);
	for (int i = 0; i < labels.rows; ++i)
		for (int j = 0; j < labels.cols; ++j)
			image.at<cv::Vec3f>(i, j) = colours[labels.at<int>(i, j)];
	return image;
}

cv::Mat BgrToLab(const cv::Mat& bgr)
{
	if (bgr.empty() || bgr.depth() != CV_8U)
		throw std::invalid_argument("BgrToLab: expected a non-empty 8-bit image");
	cv::Mat colour;
	switch (bgr.channels())
	{
	case 1:
		cv::cvtColor(bgr, colour, cv::COLOR_GRAY2BGR);
		break;
	case 3:
		colour = bgr;
		break;
	case 4:
		cv::cvtColor(bgr, colour, cv::COLOR_BGRA2BGR);
		break;
	default:
		throw std::invalid_argument("BgrToLab: expected a 1-, 3- or 4-channel image");
	}
	// OpenCV's BGR2Lab expects BGR data (imread's order); from floats in
	// [0, 1] it gives L in [0, 100] and a, b in about [-127, 127]
	cv::Mat scaled, lab;
	colour.convertTo(scaled, CV_32F, 1.0 / 255.0);
	cv::cvtColor(scaled, lab, cv::COLOR_BGR2Lab);
	return lab;
}

cv::Mat LabToBgr(const cv::Mat& lab)
{
	CheckLabImage(lab, "LabToBgr");
	cv::Mat scaled, bgr;
	cv::cvtColor(lab, scaled, cv::COLOR_Lab2BGR);
	scaled.convertTo(bgr, CV_8U, 255.0);  // rounds and saturates
	return bgr;
}

cv::Mat DrawBoundaries(const cv::Mat& bgr, const cv::Mat& labels, const cv::Vec3b& colour)
{
	if (bgr.type() != CV_8UC3 || labels.type() != CV_32S || bgr.size() != labels.size())
		throw std::invalid_argument("DrawBoundaries: expected an 8-bit BGR image and CV_32S labels of the same size");
	cv::Mat image = bgr.clone();
	for (int i = 0; i < labels.rows; ++i)
	{
		for (int j = 0; j < labels.cols; ++j)
		{
			const int label = labels.at<int>(i, j);
			const bool right = j + 1 < labels.cols && labels.at<int>(i, j + 1) != label;
			const bool below = i + 1 < labels.rows && labels.at<int>(i + 1, j) != label;
			if (right || below)
				image.at<cv::Vec3b>(i, j) = colour;
		}
	}
	return image;
}
