//
//  RLPre.cpp
//  RelationExtraction
//
//  Created by Feng Jun on 05/01/2017.
//  Copyright © 2017 Feng Jun. All rights reserved.
//

#include "RLPre.h"

namespace RLPre {
    
    vector<int> shuffleIndex;
    vector<string> b_train;
    vector<int> c_train;
    vector<int> chosenSentence;
    vector<int> allChosenSentence;
    vector<vector<int> > chosenSenSet;
    vector<double> reward;
    double score = 0;
    double score_tmp = 0, score_max = 0;
    pthread_mutex_t mutex1;
    float alpha;
    int turn;
    int ok = 1000;
    double firstScore = 1;
    double bestLoss = -1000000;
    
    void ReadData()
    {
        string tmpPath = pathString + "data/pretrain/matrixW1+B1.txt";
        FILE *fout = fopen(tmpPath.c_str(), "r");
        //FILE *fout = fopen(("/Users/fengjun/Documents/Research/relation extraction/code/RelationExtraction/out/matrixW1+B1.txt"+version).c_str(), "r");
        fscanf(fout,"%d%d%d%d", &dimensionC, &dimension, &window, &dimensionWPE);
        for (int i = 0; i < dimensionC; i++) {
            //cout<<i<<endl;
            for (int j = 0; j < dimension * window; j++)
                fscanf(fout, "%f", &matrixW1[i* dimension*window+j]);
            for (int j = 0; j < dimensionWPE * window; j++)
                fscanf(fout, "%f", &matrixW1PositionE1[i* dimensionWPE*window+j]);
            for (int j = 0; j < dimensionWPE * window; j++)
                fscanf(fout, "%f", &matrixW1PositionE2[i* dimensionWPE*window+j]);
            fscanf(fout, "%f", &matrixB1[i]);
            //            printf("%f\n", matrixB1[i]);
        }
        fclose(fout);
        
        tmpPath = pathString + "data/pretrain/matrixRl.txt";
        fout = fopen(tmpPath.c_str(), "r");
        //fout = fopen(("/Users/fengjun/Documents/Research/relation extraction/code/RelationExtraction/out/matrixRl.txt"+version).c_str(), "r");
        fscanf(fout,"%d%d", &relationTotal, &dimensionC);
        for (int i = 0; i < relationTotal; i++) {
            for (int j = 0; j < dimensionC; j++)
                fscanf(fout, "%f", &matrixRelation[i * dimensionC + j]);
        }
        for (int i = 0; i < relationTotal; i++)
            fscanf(fout, "%f", &matrixRelationPr[i]);
        fclose(fout);
        
        tmpPath = pathString + "data/pretrain/matrixPosition.txt";
        fout = fopen(tmpPath.c_str(), "r");
        //fout = fopen(("/Users/fengjun/Documents/Research/relation extraction/code/RelationExtraction/out/matrixPosition.txt"+version).c_str(), "r");
        fscanf(fout,"%d%d%d", &PositionTotalE1, &PositionTotalE2, &dimensionWPE);
        for (int i = 0; i < PositionTotalE1; i++) {
            for (int j = 0; j < dimensionWPE; j++)
                fscanf(fout, "%f", &positionVecE1[i * dimensionWPE + j]);
        }
        for (int i = 0; i < PositionTotalE2; i++) {
            for (int j = 0; j < dimensionWPE; j++)
                fscanf(fout, "%f", &positionVecE2[i * dimensionWPE + j]);
        }
        fclose(fout);
        //        cout<<71<<endl;
        
        tmpPath = pathString + "data/pretrain/word2vec.txt";
        fout = fopen(tmpPath.c_str(), "r");
        //fout = fopen(("/Users/fengjun/Documents/Research/relation extraction/code/RelationExtraction/out/word2vec.txt"+version).c_str(), "r");
        fscanf(fout,"%d%d",&wordTotal,&dimension);
        for (int i = 0; i < wordTotal; i++)
        {
            for (int j=0; j<dimension; j++)
                fscanf(fout,"%f", &wordVec[i*dimension+j]);
        }
        fclose(fout);
    }
    
    void preprocess()
    {
        matrixRelation = (float *)calloc(dimensionC * relationTotal, sizeof(float));
        matrixRelationPr = (float *)calloc(relationTotal, sizeof(float));
        matrixRelationPrDao = (float *)calloc(relationTotal, sizeof(float));
        updateMatrixRelationPr = (float *)calloc(relationTotal, sizeof(float));
        
        wordVecDao = (float *)calloc(dimension * wordTotal, sizeof(float));
        updateWordVec = (float *)calloc(dimension * wordTotal, sizeof(float));
        
        positionVecE1 = (float *)calloc(PositionTotalE1 * dimensionWPE, sizeof(float));
        positionVecE2 = (float *)calloc(PositionTotalE2 * dimensionWPE, sizeof(float));
        
        matrixW1 = (float*)calloc(dimensionC * dimension * window, sizeof(float));
        matrixW1PositionE1 = (float *)calloc(dimensionC * dimensionWPE * window, sizeof(float));
        matrixW1PositionE2 = (float *)calloc(dimensionC * dimensionWPE * window, sizeof(float));
        matrixB1 = (float*)calloc(dimensionC, sizeof(float));
        
        version = "";
        
        ReadData();
        
        featureLen = dimension * 2 + dimensionC * 2 + 1;
        featureW = (float *) calloc(featureLen, sizeof(float));
        featureWDao = (float *) calloc(featureLen, sizeof(float));
        bestFeatureW = (float *) calloc(featureLen, sizeof(float));
        
        for (int i = 0; i < featureLen; i ++)
        {
            featureList.push_back(0);
            featureW[i] = 0;
            //            featureW[i] = getRandU(-con, con);
            bestFeatureW[i] = 0;
        }
    }
    
    
    
    
   
    
    void InitFeatureList()
    {
        for (int i = 0; i < featureList.size(); i ++)
            featureList[i] = 0;
    }
    
    void UpdateFeatureLists(int flag, int id, float turn, float loss)
    {
        if (flag == 0)
        {
            /*
             for (int i = 0; i < dimension; i ++)
             featureList[i] = wordVec[headList[id] * dimension + i];
             for (int i = dimension; i < 2 * dimension; i ++)
             featureList[i] = wordVec[tailList[id] * dimension + i - dimension];
             */
            for (int i = 0; i < dimension; i ++)
                featureList[i] = entityVec[headEntityList[id] * dimension + i];
            for (int i = dimension; i < 2 * dimension; i ++)
                featureList[i] = entityVec[tailEntityList[id] * dimension + i];
            
            int jj = 0;
            for (int i = dimension * 2 + dimensionC; i < dimension * 2 + dimensionC * 2; i ++)
            {
                featureList[i] = sentenceVec[id][jj];
                jj ++;
            }
            int index = dimension * 2 + dimensionC * 2;
            //            featureList[index ++] = turn;
            //            featureList[index ++] = loss / firstScore;
            //
            featureList[index ++ ] = 1;
        }
        
        if (flag == 1)
        {
            int jj = 0;
            int senSize = chosenSentence.size();
            for (int i = dimension * 2; i < dimension * 2 + dimensionC; i ++)
            {
                if (senSize == 0)
                    break;
                featureList[i] = (featureList[i] * (senSize - 1)) + sentenceVec[chosenSentence[senSize - 1]][jj];
                featureList[i] /= senSize;
                jj ++;
            }
        }
        //                if (ok < 10)
        //                {
        //                    for (int i = 0; i < featureList.size(); i ++)
        //                        printf("%lf ", featureList[i]);
        //                    printf("\n");
        //                    ok ++;
        //                }
    }
    
    int GetAction(int id, int k, int j)
    {
        // p = P(a = 1|feature)
        double p = 0;
        for (int i = 0; i < featureList.size(); i ++)
        {
            p += featureList[i] * featureWDao[i];
            //            if (ok < 30 && turn != 0 && j == 0 && k == 0)
            //                printf("%d %d %lf %lf %lf\n", id, i, featureList[i], featureWDao[i], p);
        }
        p = sigmod(p);
        //        if (j == 0)
        //            printf("%lf ", p);
        ok ++;
        float randNum = getRandU(0, 1);
        if (randNum < p)
            return 1;
        else return 0;
    }
    
    
    int DecideAction(int id, int k)
    {
        // p = P(a = 1|feature)
        double p = 0;
        for (int i = 0; i < featureList.size(); i ++)
        {
            p += featureList[i] * featureWDao[i];
            //            if (ok < 30 && turn != 0 && j == 0 && k == 0)
            //                printf("%d %d %lf %lf %lf\n", id, i, featureList[i], featureWDao[i], p);
        }
        p = sigmod(p);
        //        printf("%lf\n", p);
        if (p >= 0.5)
            return 1;
        else return 0;
    }
    
    void selection()
    {
        matrixRelationDao = (float *)calloc(dimensionC*relationTotal, sizeof(float));
        matrixW1Dao =  (float*)calloc(dimensionC * dimension * window, sizeof(float));
        matrixB1Dao =  (float*)calloc(dimensionC, sizeof(float));
        
        positionVecDaoE1 = (float *)calloc(PositionTotalE1 * dimensionWPE, sizeof(float));
        positionVecDaoE2 = (float *)calloc(PositionTotalE2 * dimensionWPE, sizeof(float));
        matrixW1PositionE1Dao = (float *)calloc(dimensionC * dimensionWPE * window, sizeof(float));
        matrixW1PositionE2Dao = (float *)calloc(dimensionC * dimensionWPE * window, sizeof(float));
        
        updateMatrixRelation = (float *)calloc(dimensionC*relationTotal, sizeof(float));
        updateMatrixW1 =  (float*)calloc(dimensionC * dimension * window, sizeof(float));
        updateMatrixB1 =  (float*)calloc(dimensionC, sizeof(float));
        
        updatePositionVecE1 = (float *)calloc(PositionTotalE1 * dimensionWPE, sizeof(float));
        updatePositionVecE2 = (float *)calloc(PositionTotalE2 * dimensionWPE, sizeof(float));
        updateMatrixW1PositionE1 = (float *)calloc(dimensionC * dimensionWPE * window, sizeof(float));
        updateMatrixW1PositionE2 = (float *)calloc(dimensionC * dimensionWPE * window, sizeof(float));
        
        b_train.clear();
        c_train.clear();
        int tmpId = 0;
        int zeroNum = 0;
        for (map<string,vector<int> >:: iterator it = bags_train.begin(); it!=bags_train.end(); it++)
        {
            c_train.push_back(b_train.size());
            shuffleIndex.push_back(b_train.size());
            b_train.push_back(it -> first);
            if (it -> second.size() == 1)
                zeroNum ++;
        }
        printf("size = 1: %d\n", zeroNum);
        zeroNum = 0;
        for (int ii = 0; ii < shuffleIndex.size(); ii ++)
        {
            for (int k = 0; k < bags_train[b_train[ii]].size(); k ++)
            {
                int i = bags_train[b_train[ii]][k];
                //                printf("%d %d %d %d\n", i, k, headList[i], tailList[i]);
                if (headList[i] == 0 || tailList[i] == 0)
                    zeroNum ++;
                //                for (int j = 0; j < dimension; j ++)
                //                    printf("%lf ", wordVec[headList[i] * dimension + j]);
                //                for (int j = 0; j < dimension; j ++)
                //                    printf("%lf ", wordVec[tailList[i] * dimension + j]);
                //                printf("\n");
                break;
            }
        }
        printf("%d %d\n", zeroNum, shuffleIndex.size());
        if (strcmp(method.c_str(), "rlpre") == 0)
            alpha = InitialAlpha * rate / batch;
        else alpha = InitialAlpha * rate / batch;
        printf("%d\n", trainLists.size());
        
        double totAvgScore;
        if (strcmp(method.c_str(), "rlpre") == 0)
        {
            sentenceVec.clear();
            lossVec.clear();
            string tmpPath = pathString + "data/pretrain/sentenceVec.txt";
            printf("%s\n", tmpPath.c_str());
            FILE *fin = fopen(tmpPath.c_str(), "r");
            printf("%s\n", tmpPath.c_str());
            if (fin == NULL)
            {
                printf("count\n");
                RL::CountSentenceVec(pathString);
                fin = fopen(tmpPath.c_str(), "r");
            }

            for (int i = 0; i < trainLists.size(); i ++)
            {
                float *r = (float *)calloc(dimensionC, sizeof(float));
                for (int j = 0; j < dimensionC; j ++)
                {
                    float tmp;
                    fscanf(fin, "%f", &tmp);
                    r[j] = tmp;
                }
                sentenceVec.push_back(r);
            }
            fclose(fin);
            
            tmpPath = pathString + "data/pretrain/lossVec.txt";
            fin = fopen(tmpPath.c_str(), "r");
            for (int i = 0; i < trainLists.size(); i ++)
            {
                double tmp;
                fscanf(fin, "%lf", &tmp);
                lossVec.push_back(tmp);
            }
            fclose(fin);
            
            double OneNum = 0;
            double OneScore = 0;
            
            for (map<string,vector<int> >:: iterator it = bags_train.begin(); it!=bags_train.end(); it++)
            {
                if (it->second.size() != 1)
                    continue;
                int i = it->second[0];
                OneScore += lossVec[i];
                OneNum ++;
            }
            printf("%lf\n", OneScore / OneNum);
            totAvgScore = OneScore / OneNum;
        }
        
//        test::test(0);
        //        memcpy(updateFeatureW, featureW, featureLen * sizeof(float));
        memcpy(featureWDao, featureW, featureLen * sizeof(float));
        for (turn = 0; turn < trainPreTimes; turn ++)
        {
            double rlLoss = 0;
            ok = 0;
            printf("turn = %d\n", turn);
            printf("alpha = %lf\n", alpha);
            
            fprintf(logg, "turn = %d\n", turn);
            fflush(logg);
            
            memcpy(updatePositionVecE1, positionVecE1, PositionTotalE1 * dimensionWPE* sizeof(float));
            memcpy(updatePositionVecE2, positionVecE2, PositionTotalE2 * dimensionWPE* sizeof(float));
            memcpy(updateMatrixW1PositionE1, matrixW1PositionE1, dimensionC * dimensionWPE * window* sizeof(float));
            memcpy(updateMatrixW1PositionE2, matrixW1PositionE2, dimensionC * dimensionWPE * window* sizeof(float));
            memcpy(updateWordVec, wordVec, dimension * wordTotal * sizeof(float));
            
            memcpy(updateMatrixW1, matrixW1, sizeof(float) * dimensionC * dimension * window);
            memcpy(updateMatrixB1, matrixB1, sizeof(float) * dimensionC);
            memcpy(updateMatrixRelationPr, matrixRelationPr, relationTotal * sizeof(float));				//add
            memcpy(updateMatrixRelation, matrixRelation, dimensionC*relationTotal * sizeof(float));
            
            memcpy(positionVecDaoE1, updatePositionVecE1, PositionTotalE1 * dimensionWPE* sizeof(float));
            memcpy(positionVecDaoE2, updatePositionVecE2, PositionTotalE2 * dimensionWPE* sizeof(float));
            memcpy(matrixW1PositionE1Dao, updateMatrixW1PositionE1, dimensionC * dimensionWPE * window* sizeof(float));
            memcpy(matrixW1PositionE2Dao, updateMatrixW1PositionE2, dimensionC * dimensionWPE * window* sizeof(float));
            memcpy(wordVecDao, updateWordVec, dimension * wordTotal * sizeof(float));
            
            memcpy(matrixW1Dao, updateMatrixW1, sizeof(float) * dimensionC * dimension * window);
            memcpy(matrixB1Dao, updateMatrixB1, sizeof(float) * dimensionC);
            memcpy(matrixRelationPrDao, updateMatrixRelationPr, relationTotal * sizeof(float));				//add
            memcpy(matrixRelationDao, updateMatrixRelation, dimensionC*relationTotal * sizeof(float));
            //            memcpy(featureW, featureW, featureLen * sizeof(float));
            
            random_shuffle(shuffleIndex.begin(), shuffleIndex.end());
 
            //            float totAvgScore = score / trainLists.size();
            printf("finish get sentence vector\n");
            //            printf("%lf\n", score / trainLists.size());
            //            fprintf(logg, "%lf\n", score / trainLists.size());
            fflush(logg);
            //            for (int i = 0; i < trainLists.size(); i ++)
            //                printf("%d %lf\n", i, lossVec[i]);
            allChosenSentence.clear();
            ok = 0;
            for (int ii = 0; ii < shuffleIndex.size();  ii ++)
            {
                chosenSenSet.clear();
                reward.clear();
                double avgReward =0 ;
                int id = shuffleIndex[ii];
                for (int j = 0; j < sampleTimes; j ++)
                {
                    chosenSentence.clear();
                    double tmpReward = 0;
                    InitFeatureList();
                    for (int k = 0; k < bags_train[b_train[id]].size(); k ++)
                    {
                        int i = bags_train[b_train[id]][k];
                        UpdateFeatureLists(0, i, turn, score);
                        int chosenA = GetAction(i, k, j);
                        //                        printf("%d %d %d %d\n", id, j, k, chosenA);
                        if (chosenA == 1)
                        {
                            //                            printf("%d %lf\n", k, lossVec[i]);
                            chosenSentence.push_back(i);
                            tmpReward += lossVec[i];
                            //                            if (j == 0)
                            //                            {
                            //                                allChosenSentence.push_back(i);
                            //                                rlLoss += lossVec[i];
                            //                            }
                            UpdateFeatureLists(1, i, turn, score);
                        }
                    }
                    if (chosenSentence.size() == 0)
                        tmpReward = totAvgScore;
                    else tmpReward /= chosenSentence.size();
                    vector<int> tmpSet = chosenSentence;
                    chosenSenSet.push_back(tmpSet);
                    reward.push_back(tmpReward);
                    avgReward += tmpReward;
                }
                avgReward /= sampleTimes;
                for (int j = 0; j < sampleTimes; j ++)
                {
                    chosenSentence.clear();
                    InitFeatureList();
                    int l = 0;
                    
                    for (int k = 0; k < bags_train[b_train[id]].size(); k ++)
                    {
                        int i = bags_train[b_train[id]][k];
                        UpdateFeatureLists(0, i, turn, score);
                        double tmpF = 0;
                        for (int x = 0; x < featureList.size(); x ++)
                            tmpF += featureList[x] * featureWDao[x];
                        tmpF = sigmod(tmpF);
                        //                        printf("%lf %lf %d\n", reward[j], avgReward, chosenSenSet[j].size());
                        if (chosenSenSet[j].size() != 0 && i == chosenSenSet[j][l])
                        {
                            // action = 1 ,update gradients
                            for (int x = 0; x < featureList.size(); x ++)
                            {
                                featureW[x] += alpha * (reward[j] - avgReward) * (1 - tmpF) * featureList[x];
                                //                                if (x == featureLen - 1)
                                //                                    printf("%d %lf %lf %lf %lf %lf %lf\n", j, featureW[x], reward[j], avgReward, reward[j] - avgReward, 1 - tmpF, featureList[x]);
                                
                                //                                printf("%lf\n", featureW[x]);
                            }
                            chosenSentence.push_back(i);
                            UpdateFeatureLists(1, i, turn, score);
                            l ++;
                        }
                        else {
                            for (int x = 0; x < featureList.size(); x ++)
                            {
                                featureW[x] += alpha * (reward[j] - avgReward) * (-1 * tmpF) * featureList[x];
                                //                                if (x == featureLen - 1)
                                //                                    printf("%d %lf %lf %lf %lf %lf %lf\n", j, featureW[x], reward[j], avgReward, reward[j] - avgReward, - tmpF, featureList[x]);
                            }
                        }
                        //                        printf("%lf\n", featureW[featureLen - 1]);
                    }
                }
                
                InitFeatureList();
                for (int k = 0; k < bags_train[b_train[id]].size(); k ++)
                {
                    int i = bags_train[b_train[id]][k];
                    
                    UpdateFeatureLists(0, i, turn, score);
                    int chosenA = DecideAction(i, k);
                    //                    printf("%d\n", chosenA);
                    if (chosenA == 1)
                    {
                        allChosenSentence.push_back(i);
                        rlLoss += lossVec[i];
                        UpdateFeatureLists(1, i, turn, score);
                        
                        //                        printf("%d %d\n", chosenA, allChosenSentence.size());
                    }
                }
                
                
            }
            
            memcpy(featureWDao, featureW, featureLen * sizeof(float));
//            string outPath = outString + "_bestRL.txt" + Int_to_String(turn);
//            FILE *fout = fopen(outPath.c_str(), "w");
//            
//            for (int i = 0; i < featureLen; i ++)
//                fprintf(fout, "%lf ", featureW[i]);
//            fprintf(fout, "\n");
//            fclose(fout);
//            
//            for (int i = 0; i < featureLen; i ++)
//                printf("%lf ", featureW[i]);
//            printf("\n");
            
            rlLoss /= allChosenSentence.size();
            if (rlLoss > bestLoss)
            {
                bestLoss = rlLoss;
                memcpy(bestFeatureW, featureW, featureLen * sizeof(float));
            }
//            fprintf(logg, "turn = %d\n", turn);
            fprintf(logg, "chosen sentence size = %d %lf %lf\n", allChosenSentence.size(), rlLoss, bestLoss);
            fflush(logg);
            printf("chosen sentence size = %d %lf %lf\n", allChosenSentence.size(), rlLoss, bestLoss);
           
        }
        
        string outPath = pathString + "data/pretrain/pre_bestRL.txt";
        FILE *fout = fopen(outPath.c_str(), "w");
        for (int i = 0; i < featureLen; i ++)
            fprintf(fout, "%lf\n", bestFeatureW[i]);
        fclose(fout);
        
        for (int i = 0; i < featureLen; i ++)
            printf("%lf", bestFeatureW[i]);
        printf("\n");
    }
    
    void beginTrain()
    {
        string tmpPath = outString + "log.txt";
        logg = fopen(tmpPath.c_str(), "w");
        
        tmpPath = outString + "pr.txt";
        prlog = fopen(tmpPath.c_str(), "w");
        
        init();
        preprocess();
        printf("finish preprocess\n");
        selection();
        fclose(logg);
        fclose(prlog);
    }
    
}

