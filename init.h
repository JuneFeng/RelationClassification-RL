//
//  init.h
//  RelationExtraction
//
//  Created by Feng Jun on 06/12/2016.
//  Copyright © 2016 Feng Jun. All rights reserved.
//

#ifndef init_h
#define init_h

#include <cstring>
#include <cstdlib>
#include <vector>
#include <map>
#include <string>
#include <float.h>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <set>
#include <algorithm>
#include <sstream>
#include <pthread.h>

#include<assert.h>
#include<ctime>
#include<sys/time.h>
#include<sstream>

using namespace std;

extern string localPathString;
extern string serverPathString;
extern string pathString;
extern string outString;

extern string version;
extern int output_model;

extern int num_threads;
extern int trainTimes;
extern int sampleTimes;
extern float InitialAlpha;
extern float reduce;
extern int tt,tt1;
extern int dimensionC;//1000;
extern int dimensionWPE;//25;
extern int window;
extern int limit;
extern float marginPositive;
extern float marginNegative;
extern float margin;
extern float Belt;
extern float*matrixB1, *matrixRelation, *matrixW1, *matrixRelationDao, *matrixRelationPr, *matrixRelationPrDao;
extern float *updateMatrixB1, *updateMatrixRelation, *updateMatrixW1, *updateMatrixRelationPr;
extern float *matrixB1_egs, *matrixRelation_egs, *matrixW1_egs, *matrixRelationPr_egs;
extern float *matrixB1_exs, *matrixRelation_exs, *matrixW1_exs, *matrixRelationPr_exs;
extern float *wordVecDao,*wordVec_egs,*wordVec_exs;
extern float *updateWordVec;
extern float *positionVecE1, *positionVecE2, *matrixW1PositionE1, *matrixW1PositionE2;
extern float *updatePositionVecE1, *updatePositionVecE2, *updateMatrixW1PositionE1, *updateMatrixW1PositionE2;
extern float *positionVecE1_egs, *positionVecE2_egs, *matrixW1PositionE1_egs, *matrixW1PositionE2_egs, *positionVecE1_exs, *positionVecE2_exs, *matrixW1PositionE1_exs, *matrixW1PositionE2_exs;
extern float *matrixW1PositionE1Dao;
extern float *matrixW1PositionE2Dao;
extern float *positionVecDaoE1;
extern float *positionVecDaoE2;
extern float *matrixW1Dao;
extern float *matrixB1Dao;
extern double mx;
extern int batch;
extern int npoch;
extern int len;
extern float rate;
extern FILE *logg;
extern FILE *prlog;
extern float eps;
extern float freezeRatio;

extern float*wordVec;
extern int wordTotal, dimension, relationTotal;
extern int PositionMinE1, PositionMaxE1, PositionTotalE1,PositionMinE2, PositionMaxE2, PositionTotalE2;
extern map<string,int> wordMapping;
extern vector<string> wordList;
extern map<string,int> relationMapping;
extern vector<int *> trainLists, trainPositionE1, trainPositionE2;
extern vector<int> trainLength;
extern vector<int> headList, tailList, relationList;
extern vector<int *> testtrainLists, testPositionE1, testPositionE2;
extern vector<int> testtrainLength;
extern vector<int> testheadList, testtailList, testrelationList;
extern vector<std::string> nam;

extern map<string,vector<int> > bags_train, bags_test;
extern vector<float> featureList;
extern float* featureW;
extern float* bestFeatureW;
extern float* featureWDao;

extern vector<float *> sentenceVec;
extern vector<double> lossVec;
extern int featureLen;

extern string method;

extern int nowTurn;

extern vector<int> headEntityList, tailEntityList;
extern vector<float> entityVec;
extern map<string, int> entityMapping;

extern float *bestMatrixB1, *bestMatrixRelation, *bestMatrixW1, *bestMatrixRelationPr;
extern float *bestPositionVecE1, *bestPositionVecE2, *bestMatrixW1PositionE1, *bestMatrixW1PositionE2;
extern float *bestWordVec;

void init();

float CalcTanh(float con);
float tanhDao(float con);
float sigmod(float con);
int getRand(int l,int r);

float getRandU(float l, float r);

void norm(float* a, int ll, int rr);

string Int_to_String(int n);


#endif /* init_h */
