/*
Kode ini dibuat dalam bahasa C++ dikarenakan implementasi
EM Clustering termasuk memiliki komputasi yang berat.
Sehingga dengan kecepatan kompuasi C++ ini, impelementasi
EM Clustering menjadi lbh cepat (execution time) jika dibandingkan menggunakan Python.
Di kode ini dugunakan libary OpenCV, yg digunakan untuk proses rescaling image MNIST
dari 28x28 ke 20x20. Bagi yg blm familiar dengan cara menambahkan library OpenCV, dapat melihat di
file README.md yang ada di dalam repository github mengenai buku ini.
*/

//memanggil header yang dibutuhkan
#include "stdafx.h"
#include <fstream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
#include <iostream>
#include <random>

using namespace std;

void read_Mnist_Label(string filename, vector<double> &vec);
void read_Mnist(string filename, vector<cv::Mat> &vec);
void read_Mnist(string filename, vector<vector<double>> &vec);
void preprocessMNIST(vector<cv::Mat> &vecOri, vector<cv::Mat> &vecResult);
void initializeMixingCoef(vector<double> &mixingCoefInit);
void initalizeParams(vector<vector<double>> &paramsInit);
void binningData(vector<vector<double>> &result, vector<cv::Mat>inputImage);
vector<vector<double>> calculateJointProb(vector<vector<double>> &binnedImageData, vector<double> mixingCoefInit, vector<vector<double>> paramsInit);
vector<vector<double>> calculateResponsibility(vector<vector<double>> jointProb);
vector<vector<double>> updateParams(vector<vector<double>> &binnedDataImage, vector<vector<double>> &responsibility);
vector<double> updateMixingCoef(vector<vector<double>> &responsibility);
vector<double> vecLabel;
vector<vector<int>> evaluateClusterResult(vector<vector<double>> &jointProb);
double calculateDeltaParams(vector<vector<double>> &currentParams, vector<vector<double>> &prevParams);
void printConfussionMatrix(map<string, int> &hastableClusterMembers, vector<vector<int>> &clusterResult);
map<string, int> evaluateMemberPerCluster(vector<vector<int>> &clusterResult, int index);

//global variable
vector<vector<double>> binnedImageData;
map<string, int> hastableLabel;
vector<vector<int>> clusterResult;//row = clusters, cols = mermbers in unordered condition

int main()
{
	cout << "Membaca dataset MNIST..." << endl;
	//membaca data gambar MNIST ke vector bertipe Mat OpenCV
	string filenameImageTest = "dataset/t10k-images.idx3-ubyte";
	vector<cv::Mat> vecImageTest;
	read_Mnist(filenameImageTest, vecImageTest);
	cout << "banyaknya gambar: " << vecImageTest.size() << endl;
	cv::waitKey(1);

	//membaca data label MNISTke dalam vector bertipe double
	string filenameLabelTest = "dataset/t10k-labels.idx1-ubyte";
	int number_of_images = 10000;
	read_Mnist_Label(filenameLabelTest, vecLabel);
	cout << "banyaknya label: " << vecLabel.size() << endl;

	//melakukan pre-processing data gambar, yakni
	//meresize gambar asli berukuran 28x28 ke 20x20
	vector<cv::Mat> vecImageTestPreprocessed;
	preprocessMNIST(vecImageTest, vecImageTestPreprocessed);
	//melakukan binning dari nilai intensitas gambarnya
	// jika >= 130 ubah ke 1, dan < 130 ubah ke 0
	binningData(binnedImageData, vecImageTestPreprocessed);
	//menginisialisasi mixing coefficient and parameter
	vector<double> mixingCoefInit;
	initializeMixingCoef(mixingCoefInit);
	vector<vector<double>> paramsInit;
	initalizeParams(paramsInit);

	//kode utama EM klastering untuk MNIST digit number
	int maxIter = 500; float tolerance = 0.0001;
	for (int a = 0; a < maxIter; a++)
	{
		//fungsi untuk menghitung join probability Bernoulli dari tiap file gambar digit
		vector<vector<double>> jointProb;
		jointProb = calculateJointProb(binnedImageData, mixingCoefInit, paramsInit);
		//fungsi untuk mengevaluasi hasil klaster, hasil = vector 2D dg row=label klaster, col=anggota klaster
		clusterResult = evaluateClusterResult(jointProb);
		map<string, int> hastableClusterMembers;
		//fungsi untuk menampilkan anggota tiap klaster yg sudah diurutkan
		hastableClusterMembers = evaluateMemberPerCluster(clusterResult, a+1);
		//hitung bobot/resposibility semua obyek data untuk tiap klaster
		vector<vector<double>> responsibility;
		responsibility = calculateResponsibility(jointProb);
		//hitung parameter Bernoulli dengan bobot/resposibility yang baru
		vector<vector<double>> params;
		params = updateParams(binnedImageData, responsibility);
		//check jika kondisi konvergen terpenuhi
		double delta = calculateDeltaParams(params, paramsInit);
		if (delta < tolerance || a == (maxIter - 1))
		{
			cout << endl << endl << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl;
			cout << "Converged! Print confusion matrix." << endl;
			printConfussionMatrix(hastableClusterMembers, clusterResult);
			break;
		}
		cout << "delta params: " << delta << endl;
		//mengupdate mixing coefficient dan parameter
		vector<double> mixingCoef;
		mixingCoef = updateMixingCoef(responsibility);
		mixingCoefInit = mixingCoef; paramsInit = params;
	}

	getchar();
	cv::waitKey();
	return 0;
}

void printConfussionMatrix(map<string, int> &hastableClusterMembers, vector<vector<int>> &clusterResult)
{
	vector<int> vecCol;
	vector<vector<int>> result;//hasil akhir klaster, row = indeks untuk nomor klaster, col = indeks untuk nomor gambar
	//assign hashable ke vector 2D
	for (int a = 0; a < 10; a++)
	{
		result.push_back(vecCol);
		for (int b = 0 ; b < 10; b++)
		{
			string label = "cluster" + to_string(a + 1) + "_" + to_string(b);//number of cluster a, handwritten b
			result.at(a).push_back(hastableClusterMembers[label]);
		}
	}
	//print confusion matrik
	float specificitySum = 0, sensitifitySum = 0;
	for (int a = 0; a < 10; a++)
	{
		int tempMax = result.at(0).at(a); int index = 0; int FP = 0;
		cout << "# Digit " << a << endl;
		cout << "     ";
		cout << endl << "D" << a << "   |   ";
		for (int b = 0; b < 10; b++)
		{
			if (result.at(b).at(a) > tempMax)
			{
				index = b;
				tempMax = result.at(b).at(a);
				//cout << endl << "IF --> a: " << a << ", b: " << b << ", index: " << index << ", tempMax: " << tempMax << endl;
			}
			cout << "C" << b+1 << ": " << result.at(b).at(a) << "; ";
		}
		cout << endl << "!D" << a << "  |   ";
		for (int b = 0; b < 10; b++)
		{
			if (b == index) { FP = clusterResult.at(b).size() - result.at(b).at(a); }
			cout << "C" << b+1 << ": " << clusterResult.at(b).size() - result.at(b).at(a) << "; ";
		}
		float sensitifity = (float) tempMax / (float) hastableLabel[to_string(a)]; 
		cout << endl << "Sensitifity : TP/Actual yes: " << tempMax <<"/"<< (float)hastableLabel[to_string(a)] << ": " << sensitifity << endl;
		float specificity = (float) ((float)(binnedImageData.size() - hastableLabel[to_string(a)]) - FP) / (float)(binnedImageData.size() - hastableLabel[to_string(a)]);
		cout << "Specificity: TN/Actual no: " << (float)(binnedImageData.size() - FP) << "/" << (float)(binnedImageData.size() - hastableLabel[to_string(a)]) << ": " << specificity << endl;
		cout << "Digit " << a << " is clustered into cluster " << index + 1 << endl;
		cout << "*************************************" << endl << endl;
		sensitifitySum += sensitifity;
		specificitySum += specificity;
	}
	cout << "average sensitifity: " << sensitifitySum / 10 << endl;
	cout << "average specificity: " << specificitySum / 10 << endl;
}

double calculateDeltaParams(vector<vector<double>> &currentParams, vector<vector<double>> &prevParams)
{
	double num = 0.0, denum = currentParams.size() * currentParams.at(0).size();
	for (int a = 0; a < currentParams.size(); a++)
	{
		for (int b = 0; b < currentParams.at(0).size(); b++)
		{
			num = num + abs(currentParams.at(a).at(b) - prevParams.at(a).at(b));
		}
	}
	double delta = num / denum;
	return delta;
}

map<string, int> evaluateMemberPerCluster(vector<vector<int>> &clusterResult, int index)
{
	map<string, int> memberPerCluster;
	//hitung keanggotaan setiap klaster
	cout << "Iteration " << index << endl;
	for (int a = 0; a < 10; a++)//looping untuk semua klaster
	{
		for (int b = 0; b < clusterResult.at(a).size(); b++)//clusterResult(k kluster, member)
		{
			string label = "cluster" + to_string(a + 1) + "_" + to_string(clusterResult.at(a).at(b));
			memberPerCluster[label] = memberPerCluster[label] + 1;
		}
	}
	//print member per klaster
	for (int a = 0; a < 10; a++)//looping untuk semua klaster
	{
		cout << "cluster " << a + 1 << ": ";
		for (int b = 0; b < 10; b++)//looping untuk semua label
		{
			string label = "cluster" + to_string(a + 1) + "_" + to_string(b);
			cout << "" << b << ": " << memberPerCluster[label] << "; ";
		}
		cout << endl;
	}
	cout << endl << "-----------------------------------" << endl;
	return memberPerCluster;
}

vector<vector<int>> evaluateClusterResult(vector<vector<double>> &jointProb)
{
	vector<int> clusterPred; vector<vector<int>> clusterResult; vector<int> vecCols;
	for (int a = 0; a < 10; a++)
	{
		clusterResult.push_back(vecCols);
	}

	for (int a = 0; a < binnedImageData.size(); a++)
	{
		int index = 0;
		double maxi = jointProb.at(0).at(a);
		for (int b = 0; b < 10; b++)
		{
			if (jointProb.at(b).at(a) > maxi)
			{
				index = b;
				maxi = jointProb.at(b).at(a);
			}
		}
		clusterPred.push_back(index);//return prediksi klaster untuk setiap data
	}
	//finishing
	for (int a = 0; a < clusterPred.size(); a++)//looping untuk semua n-data
	{
		clusterResult.at(clusterPred[a]).push_back(vecLabel[a]);
	}
	return clusterResult;
}

vector<double> updateMixingCoef(vector<vector<double>> &responsibility)
{
	vector<double> result;
	for (int a = 0; a < 10; a++)//looping untuk semua klaster
	{
		double num = 0.0;
		for (int b = 0; b < binnedImageData.size(); b++)//looping untuk semua n-data
		{
			num = num + responsibility.at(a).at(b);
		}
		double mixingCoefTemp = num / binnedImageData.size();
		result.push_back(mixingCoefTemp);
	}
	return result;
}

vector<vector<double>> updateParams(vector<vector<double>> &binnedDataImage, vector<vector<double>> &responsibility)
{
	vector<vector<double>> params; vector<double> vecCols;
	for (int a = 0; a < 10; a++)//looping untuk ke 10 klaster
	{
		params.push_back(vecCols);
		for (int b = 0; b < binnedDataImage.at(0).size(); b++)//looping untuk semua pixel
		{	
			double num = 0.0; double denum = 0.0;
			for (int c = 0; c < binnedDataImage.size(); c++)//loop untuk semua n-data
			{
				num = num + (responsibility.at(a).at(c) * binnedDataImage.at(c).at(b));
				denum = denum + responsibility.at(a).at(c);
			}
			double paramsTemp = num / denum;
			params.at(a).push_back(paramsTemp);
		}
	}
	return params;
}

vector<vector<double>> calculateResponsibility(vector<vector<double>> jointProb)
{
	vector<vector<double>> responsibility; vector<double> vecCols;
	for (int a = 0; a < 10; a++){ responsibility.push_back(vecCols); }
	for (int a = 0; a < jointProb.at(0).size(); a++)//looping untuk semua n-data
	{
		double denum = 0.0;
		for (int b = 0; b < 10; b++)//looping untuk 10 klaster
		{
			denum = denum + jointProb.at(b).at(a);
		}
		for (int c = 0; c < 10; c++)
		{
			double responsibilityTemp = jointProb.at(c).at(a)/denum;
			responsibility.at(c).push_back(responsibilityTemp);
		}
	}
	return responsibility;
}

void binningData(vector<vector<double>> &result, vector<cv::Mat>inputImage)
{
	vector<double> vecCol;
	for (int a = 0; a < inputImage.size(); a++)//looping untuk semua images
	{
		result.push_back(vecCol);
		for (int b = 0; b < inputImage[0].rows; b++)//looping untuk semua baris
		{
			vector<double> temp;
			inputImage[a].row(b).copyTo(temp);
			for (int c = 0; c < temp.size(); c++)
			{
				result.at(a).push_back(floor(temp[c]/130));
			}
		}
	}
}

vector<vector<double>> calculateJointProb(vector<vector<double>> &binnedImageData, vector<double> mixingCoefInit, vector<vector<double>> paramsInit)
{
	vector<vector<double>> jointProbResult; vector<double> vecCols;
	//hitung join probability
	for (int a = 0; a < 10; a++)
	{
		jointProbResult.push_back(vecCols);
	}
	for (int a=0; a<10; a++)//looping untuk semua klaster
	{
		for (int b=0; b<binnedImageData.size(); b++)//looping untuk semua images
		{	
			double jointProbTemp = 1;
			for (int c = 0; c < binnedImageData.at(0).size(); c++)//looping untuk semua pixels
			{
				double param = paramsInit.at(a).at(c);
				int x = binnedImageData.at(b).at(c);
				double bernoulliProb = pow(param, x) * pow((1-param), (1 - x));
				jointProbTemp = jointProbTemp * bernoulliProb;
			}
			jointProbTemp = jointProbTemp * mixingCoefInit[a];
			jointProbResult.at(a).push_back(jointProbTemp);
		}
	}
	return jointProbResult;
}

void initalizeParams(vector<vector<double>> &paramsInit)
{
	vector<double> vecCol;
	const float range_from = 0.3;
	const float range_to = 0.8;
	std::random_device rand_dev;
	std::mt19937 generator(rand_dev());
	std::uniform_real_distribution<double> distr(range_from, range_to);

	for (int a = 0; a<10; a++)
	{
		paramsInit.push_back(vecCol);
		for (int b = 0; b < 784; b++)
		{
			paramsInit.at(a).push_back(distr(generator));
		}
	}
}

void initializeMixingCoef(vector<double> &mixingCoefInit)
{
	for (int a = 0; a < 10; a++)
	{
		mixingCoefInit.push_back(1.0 / 10.0);
	}
}
void preprocessMNIST(vector<cv::Mat> &vecOri, vector<cv::Mat> &vecResult)
{
	for (int a = 0; a < vecOri.size(); a++)
	{
		int startX = 4, startY = 4, width = 20, height = 20;
		cv::Mat ROI(vecOri[a], cv::Rect(startX, startY, width, height));
		cv::Size size(13, 13);
		resize(ROI, ROI, size);
		vecResult.push_back(ROI);
	}
}

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist(string filename, vector<vector<double> > &vec)
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		for (int i = 0; i < number_of_images; ++i)
		{
			vector<double> tp;
			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					tp.push_back((double)temp);
				}
			}
			vec.push_back(tp);
		}
	}
}

void read_Mnist(string filename, vector<cv::Mat> &vec) {
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		for (int i = 0; i < number_of_images; ++i)
		{
			cv::Mat tp = cv::Mat::zeros(n_rows, n_cols, CV_8UC1);
			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					tp.at<uchar>(r, c) = (int)temp;
				}
			}
			vec.push_back(tp);
		}
	}
}


void read_Mnist_Label(string filename, vector<double> &vec)
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		for (int i = 0; i < number_of_images; ++i)
		{
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			vec.push_back((double)temp);
			hastableLabel[to_string(temp)] += 1;
		}
	}
}
