//
//  CNNSet.cpp
//  RelationExtraction
//
//  Created by Feng Jun on 19/12/2016.
//  Copyright © 2016 Feng Jun. All rights reserved.
//

#include "RL.h"

namespace RL {
    
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
    int threadLen = 1;
    
    void UpdateValue(float *a, float *b, int len)
    {
        for (int i = 0; i < len; i ++)
            a[i] = freezeRatio * b[i] + (1 - freezeRatio) * a[i];
    }
    
    void preprocess()
    {
        matrixRelation = (float *)calloc(dimensionC * relationTotal, sizeof(float));
        matrixRelationPr = (float *)calloc(relationTotal, sizeof(float));
        matrixRelationPrDao = (float *)calloc(relationTotal, sizeof(float));
        updateMatrixRelationPr = (float *)calloc(relationTotal, sizeof(float));
        bestMatrixRelationPr = (float *)calloc(relationTotal, sizeof(float));
        
        wordVecDao = (float *)calloc(dimension * wordTotal, sizeof(float));
        updateWordVec = (float *)calloc(dimension * wordTotal, sizeof(float));
        bestWordVec = (float *)calloc(dimension * wordTotal, sizeof(float));
        
        positionVecE1 = (float *)calloc(PositionTotalE1 * dimensionWPE, sizeof(float));
        positionVecE2 = (float *)calloc(PositionTotalE2 * dimensionWPE, sizeof(float));
        
        matrixW1 = (float*)calloc(dimensionC * dimension * window, sizeof(float));
        matrixW1PositionE1 = (float *)calloc(dimensionC * dimensionWPE * window, sizeof(float));
        matrixW1PositionE2 = (float *)calloc(dimensionC * dimensionWPE * window, sizeof(float));
        matrixB1 = (float*)calloc(dimensionC, sizeof(float));
        
        version = "";
        
        string tmpPath = pathString + "data/pretrain/matrixW1+B1.txt";
        FILE *fout = fopen(tmpPath.c_str(), "r");
        //FILE *fout = fopen(("/Users/fengjun/Documents/Research/relation extraction/code/RelationExtraction/out/matrixW1+B1.txt"+version).c_str(), "r");
        fscanf(fout,"%d%d%d%d", &dimensionC, &dimension, &window, &dimensionWPE);
        for (int i = 0; i < dimensionC; i++) {
            //cout<<i<<endl;
            for (int j = 0; j < dimension * window; j ++)
                fscanf(fout, "%f", &matrixW1[i * dimension * window + j]);
            for (int j = 0; j < dimensionWPE * window; j ++)
                fscanf(fout, "%f", &matrixW1PositionE1[i * dimensionWPE * window + j]);
            for (int j = 0; j < dimensionWPE * window; j++)
                fscanf(fout, "%f", &matrixW1PositionE2[i * dimensionWPE * window + j]);
            fscanf(fout, "%f", &matrixB1[i]);
            //            printf("%lf\n", matrixB1[i]);
        }
        fclose(fout);
        
        tmpPath = pathString + "data/pretrain/matrixRl.txt";
        fout = fopen(tmpPath.c_str(), "r");
        //fout = fopen(("/Users/fengjun/Documents/Research/relation extraction/code/RelationExtraction/out/matrixRl.txt"+version).c_str(), "r");
        fscanf(fout,"%d%d", &relationTotal, &dimensionC);
        for (int i = 0; i < relationTotal; i ++) {
            for (int j = 0; j < dimensionC; j++)
                fscanf(fout, "%f", &matrixRelation[i * dimensionC + j]);
        }
        for (int i = 0; i < relationTotal; i ++)
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
        cout<<71<<endl;
        
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
        
        
        featureLen = dimension * 2 + dimensionC * 2 + 1;
        featureW = (float *) calloc(featureLen, sizeof(float));
        featureWDao = (float *) calloc(featureLen, sizeof(float));
        bestFeatureW = (float *) calloc(featureLen, sizeof(float));
        
        featureList.clear();
        if (strcmp(method.c_str(), "rl") == 0)
        {
            string tmpPath = pathString + "data/pretrain/pre_bestRL.txt";
            FILE *fout = fopen(tmpPath.c_str(), "r");
            for (int i = 0; i < featureLen; i ++)
            {
                featureList.push_back(0);
                fscanf(fout, "%f", &featureW[i]);
            }
        }
        
        sentenceVec.clear();
        lossVec.clear();
        sentenceVec.clear();
        lossVec.clear();
//        tmpPath = pathString + "/data/pretrain/sentenceVec.txt";
//        FILE *fin = fopen(tmpPath.c_str(), "r");
//        
//        for (int i = 0; i < trainLists.size(); i ++)
//        {
//            float *r = (float *)calloc(dimensionC, sizeof(float));
//            for (int j = 0; j < dimensionC; j ++)
//            {
//                float tmp;
//                fscanf(fin, "%f", &tmp);
//                r[j] = tmp;
////                lossVec.push_back(0);
//            }
//            sentenceVec.push_back(r);
//        }
//        fclose(fin);
//        
//        tmpPath = pathString + "/data/pretrain/lossVec.txt";
//        fin = fopen(tmpPath.c_str(), "r");
//        score = 0;
//        for (int i = 0; i < trainLists.size(); i ++)
//        {
//            double tmp;
//            fscanf(fin, "%lf", &tmp);
//            lossVec.push_back(tmp);
//            score += tmp;
//        }
//        fclose(fin);
    

        for (int i = 0; i < trainLists.size(); i ++)
        {
            float *tmp = (float *) calloc(dimensionC, sizeof(float));
            sentenceVec.push_back(tmp);
            lossVec.push_back(0);
        }
    }
    
    
    double GetSentenceVec(int flag, int *sentence, int *trainPositionE1, int *trainPositionE2, int len, int e1, int e2, int r1, double &res, double &res1, float *matrixW1, float *matrixB1, float *r, float *matrixRelation,
                          float *positionVecE1, float *positionVecE2, float*matrixW1PositionE1, float*matrixW1PositionE2,  float *matrixRelationPr, float alpha)
    {
        int tip[dimensionC];
        
        for (int i = 0; i < dimensionC; i++) {
            int last = i * dimension * window;
            int lastt = i * dimensionWPE * window;
            float mx = -FLT_MAX;
            for (int i1 = 0; i1 <= len - window; i1++) {
                double res = 0;
                int tot = 0;
                int tot1 = 0;
                for (int j = i1; j < i1 + window; j++)
                    if (j>=0&&j<len){
                        int last1 = sentence[j] * dimension;
                        for (int k = 0; k < dimension; k++) {
                            res += matrixW1[last + tot] * wordVec[last1+k];
                            tot++;
                        }
                        int last2 = trainPositionE1[j] * dimensionWPE;
                        int last3 = trainPositionE2[j] * dimensionWPE;
                        for (int k = 0; k < dimensionWPE; k++) {
                            res += matrixW1PositionE1[lastt + tot1] * positionVecE1[last2+k];
                            res += matrixW1PositionE2[lastt + tot1] * positionVecE2[last3+k];
                            tot1++;
                        }
                    }
                    else
                    {
                        tot+=dimension;
                        tot1+=dimensionWPE;
                    }
                if (res > mx) {
                    mx = res;
                    tip[i] = i1;
                }
            }
            r[i] = mx + matrixB1[i];
        }
        
        for (int i = 0; i < dimensionC; i++) {
            r[i] = CalcTanh(r[i]);
        }
        //        if (flag == 0)
        //            sentenceVec.push_back(r);
        
        //        for (int i = 0; i < dimensionC; i ++)
        //            printf("%lf ", r[i]);
        //        printf("\n");
        
        vector<int> dropout;
        for (int i = 0; i < dimensionC; i++)
            if (flag == 0)
                dropout.push_back(1);
            else dropout.push_back(rand()%2);
        
        vector<double> f_r;
        double sum = 0;
        for (int j = 0; j < relationTotal; j++) {
            double s = 0;
            for (int i = 0; i < dimensionC; i++) {
                s += dropout[i] * r[i] * matrixRelation[j * dimensionC + i];
            }
            s += matrixRelationPr[j];
            f_r.push_back(exp(s));
            sum+=f_r[j];
        }
        double rt = log(f_r[r1]) - log(sum);
        //        if (flag == 0)
        //        {
        //            lossVec.push_back(rt);
        //            score += rt;
        //        }
        
        if (flag)
        {
            float s1, g, s2;
            for (int i = 0; i < dimensionC; i++) {
                if (dropout[i]==0)
                    continue;
                int last = i * dimension * window;
                int tot = 0;
                int lastt = i * dimensionWPE * window;
                int tot1 = 0;
                float g1 = 0;
                for (int r2 = 0; r2<relationTotal; r2++)
                {
                    g = f_r[r2]/(sum + eps) *alpha;//sigmod(margin*(marginNegative + s2)) * margin * alpha;
                    if (r2 == r1)
                        g -= alpha;
                    g1 += g * matrixRelation[r2 * dimensionC + i] * (1 -  r[i] * r[i]);
                    updateMatrixRelation[r2 * dimensionC + i] -= g * r[i];
                    if (i==0)
                        updateMatrixRelationPr[r2] -= g;
                }
                for (int j = 0; j < window; j++)
                    if (tip[i]+j>=0&&tip[i]+j<len){
                        int last1 = sentence[tip[i] + j] * dimension;
                        for (int k = 0; k < dimension; k++) {
                            updateMatrixW1[last + tot] -= g1 * wordVec[last1+k];
                            updateWordVec[last1 + k] -= g1 * matrixW1[last + tot];
                            tot++;
                        }
                        int last2 = trainPositionE1[tip[i] + j] * dimensionWPE;
                        int last3 = trainPositionE2[tip[i] + j] * dimensionWPE;
                        for (int k = 0; k < dimensionWPE; k++) {
                            updateMatrixW1PositionE1[lastt + tot1] -= g1 * positionVecE1[last2 + k];
                            updateMatrixW1PositionE2[lastt + tot1] -= g1 * positionVecE2[last3 + k];
                            updatePositionVecE1[last2 + k] -= g1 * matrixW1PositionE1[lastt + tot1];
                            updatePositionVecE2[last3 + k] -= g1 * matrixW1PositionE2[lastt + tot1];
                            tot1++;
                        }
                    }
                updateMatrixB1[i] -= g1;
            }
            
            for (int i = 0; i < dimensionC; i++) {
                int last = dimension * window * i;
                res1+=Belt * matrixB1[i] * matrixB1[i];
                
                for (int j = dimension * window -1; j >= 0; j--) {
                    res1+= Belt * matrixW1[last + j] * matrixW1[last + j];
                    updateMatrixW1[last + j] += - Belt * matrixW1[last + j] * alpha * 2;
                }
                
                last = dimensionWPE * window * i;
                for (int j = dimensionWPE * window -1; j>=0; j--) {
                    updateMatrixW1PositionE1[last + j] += -Belt * matrixW1PositionE1[last + j] * alpha * 2;
                    updateMatrixW1PositionE2[last + j] += -Belt * matrixW1PositionE2[last + j] * alpha * 2;
                }
                
                updateMatrixB1[i] += -Belt * matrixB1[i] *alpha * 2;
            }
        }
        return rt;
    }
    
    void* trainMode(void *id )
    {
        unsigned long long next_random = (long long)id;
        float *r = (float *)calloc(dimensionC, sizeof(float));
        {
            double res = 0;
            double res1 = 0;
            for (int k1 = batch; k1 > 0; k1--)
            {
                int i = getRand(0, allChosenSentence.size());
                //                printf("%d %d\n", i, allChosenSentence.size());
                i = allChosenSentence[i];
                score+= GetSentenceVec(1,trainLists[i], trainPositionE1[i], trainPositionE2[i], trainLength[i], headList[i], tailList[i], relationList[i], res, res1, matrixW1Dao, matrixB1Dao, r, matrixRelationDao, positionVecDaoE1, positionVecDaoE2, matrixW1PositionE1Dao, matrixW1PositionE2Dao, matrixRelationPrDao, alpha);
            }
        }
        free(r);
        
        return NULL;
    }
    
    
    void UpdateCNN()
    {
        len = allChosenSentence.size();
        npoch = len / (batch * num_threads);
        double score1 = score;
        score = 0;
        for (int k = 1; k <= npoch; k++) {
            score_max += batch * num_threads;
            memcpy(positionVecDaoE1, updatePositionVecE1, PositionTotalE1 * dimensionWPE* sizeof(float));
            memcpy(positionVecDaoE2, updatePositionVecE2, PositionTotalE2 * dimensionWPE* sizeof(float));
            memcpy(matrixW1PositionE1Dao, updateMatrixW1PositionE1, dimensionC * dimensionWPE * window* sizeof(float));
            memcpy(matrixW1PositionE2Dao, updateMatrixW1PositionE2, dimensionC * dimensionWPE * window* sizeof(float));
            memcpy(wordVecDao, updateWordVec, dimension * wordTotal * sizeof(float));
            
            memcpy(matrixW1Dao, updateMatrixW1, sizeof(float) * dimensionC * dimension * window);
            memcpy(matrixB1Dao, updateMatrixB1, sizeof(float) * dimensionC);
            memcpy(matrixRelationPrDao, updateMatrixRelationPr, relationTotal * sizeof(float));				//add
            memcpy(matrixRelationDao, updateMatrixRelation, dimensionC*relationTotal * sizeof(float));
            
            
            
            pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
            for (int a = 0; a < num_threads; a++)
                pthread_create(&pt[a], NULL, trainMode,  (void *)a);
            for (int a = 0; a < num_threads; a++)
                pthread_join(pt[a], NULL);
            free(pt);
            if (k%(npoch/5)==0)
            {
                fprintf(logg, "npoch:\t %d %d %d\n", turn, k, npoch);
                cout<<"npoch:\t"<<turn<<'\t'<<k<<'/'<<npoch<<endl;
                fprintf(logg, "score:\t %lf %lf\n", score - score1, score_tmp);
                cout<<"score:\t"<<score-score1<<' '<<score_tmp<<endl;
                score1 = score;
                fflush(logg);
            }
            
            //            printf("id = %d\n", k);
            //            printf("update value\n");
            //            for (int i = 0; i < relationTotal; i ++)
            //                printf("%lf ", updateMatrixRelationPr[i]);
            //            printf("\n");
            //            printf("previous value\n");
            //            for (int i = 0; i < relationTotal; i ++)
            //                printf("%lf ", matrixRelationPr[i]);
            //            printf("\n");
            UpdateValue(&positionVecE1[0], &updatePositionVecE1[0], PositionTotalE1 * dimensionWPE);
            UpdateValue(&positionVecE2[0], &updatePositionVecE2[0], PositionTotalE2 * dimensionWPE);
            UpdateValue(&matrixW1PositionE1[0], &updateMatrixW1PositionE1[0], dimensionC * dimensionWPE * window);
            UpdateValue(&matrixW1PositionE2[0], &updateMatrixW1PositionE2[0], dimensionC * dimensionWPE * window);
            UpdateValue(&wordVec[0], &updateWordVec[0], dimension * wordTotal);
            
            UpdateValue(&matrixW1[0], &updateMatrixW1[0], dimensionC * dimension * window);
            UpdateValue(&matrixB1[0], &updateMatrixB1[0], dimensionC);
            UpdateValue(&matrixRelationPr[0], &updateMatrixRelationPr[0], relationTotal);				//add
            UpdateValue(&matrixRelation[0], &updateMatrixRelation[0], dimensionC*relationTotal);
            //            printf("now value\n");
            //            for (int i = 0; i < relationTotal; i ++)
            //                printf("%lf ", matrixRelationPr[i]);
            //            printf("\n");
        }
    }
    
    void InitFeatureList()
    {
        for (int i = 0; i < featureList.size(); i ++)
            featureList[i] = 0;
    }
    
    void UpdateFeatureLists(int flag, int id, float turn, double loss)
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
            //b
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
        //        if (ok < 10)
        //        {
        //            for (int i = 0; i < featureList.size(); i ++)
        //                printf("%lf ", featureList[i]);
        //            printf("\n");
        //            ok ++;
        //        }
    }
    
    int GetAction(int id, int k, int j)
    {
        // p = P(a = 1|feature)
        double p = 0;
        for (int i = 0; i < featureList.size(); i ++)
        {
            p += featureList[i] * featureW[i];
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
            p += featureList[i] * featureW[i];
            //            if (ok < 30 && turn != 0 && j == 0 && k == 0)
            //                printf("%d %d %lf %lf %lf\n", id, i, featureList[i], featureWDao[i], p);
        }
        p = sigmod(p);
        //        printf("%lf\n", p);
        if (p >= 0.5)
            return 1;
        else return 0;
    }
    
    void* ScoreModel(void *id)
    {
        int threadId = (long long) id;
        threadId *= threadLen;
        for (int i = threadId; i < threadId + threadLen; i ++)
        {
            if (i >= trainLists.size()) break;
            double res = 0;
            double res1 = 0;
            float *r = (float *)calloc(dimensionC, sizeof(float));
            {
                double tmp = GetSentenceVec(0,trainLists[i], trainPositionE1[i], trainPositionE2[i], trainLength[i], headList[i], tailList[i], relationList[i], res, res1, matrixW1, matrixB1, r, matrixRelation, positionVecE1, positionVecE2, matrixW1PositionE1, matrixW1PositionE2, matrixRelationPr, alpha);
                score += tmp;
                lossVec[i] = tmp;
                //                printf("%d %lf\n", i, tmp);
                sentenceVec[i] = r;
            }
            free(r);
            
        }
        
        return NULL;
    }
    
    void UpdateScore()
    {
        
        threadLen = trainLists.size() / num_threads;
        pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
        for (int a = 0; a < num_threads + 1; a++)
            pthread_create(&pt[a], NULL, ScoreModel,  (void *)a);
        for (int a = 0; a < num_threads + 1; a++)
            pthread_join(pt[a], NULL);
        free(pt);
        
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
        
        bestMatrixRelation = (float *)calloc(dimensionC*relationTotal, sizeof(float));
        bestMatrixW1 =  (float*)calloc(dimensionC * dimension * window, sizeof(float));
        bestMatrixB1 =  (float*)calloc(dimensionC, sizeof(float));
        
        bestPositionVecE1 = (float *)calloc(PositionTotalE1 * dimensionWPE, sizeof(float));
        bestPositionVecE2 = (float *)calloc(PositionTotalE2 * dimensionWPE, sizeof(float));
        bestMatrixW1PositionE1 = (float *)calloc(dimensionC * dimensionWPE * window, sizeof(float));
        bestMatrixW1PositionE2 = (float *)calloc(dimensionC * dimensionWPE * window, sizeof(float));
        
        b_train.clear();
        c_train.clear();
        int tmpId = 0;
        int zeroNum = 0;
        for (map<string,vector<int> >:: iterator it = bags_train.begin(); it!=bags_train.end(); it++)
        {
            c_train.push_back(b_train.size());
            shuffleIndex.push_back(b_train.size());
            b_train.push_back(it -> first);
        }
        
        alpha = InitialAlpha * rate / batch;
        printf("%d\n", trainLists.size());
        
        double totAvgScore;
        
        memcpy(updatePositionVecE1, positionVecE1, PositionTotalE1 * dimensionWPE* sizeof(float));
        memcpy(updatePositionVecE2, positionVecE2, PositionTotalE2 * dimensionWPE* sizeof(float));
        memcpy(updateMatrixW1PositionE1, matrixW1PositionE1, dimensionC * dimensionWPE * window* sizeof(float));
        memcpy(updateMatrixW1PositionE2, matrixW1PositionE2, dimensionC * dimensionWPE * window* sizeof(float));
        memcpy(updateWordVec, wordVec, dimension * wordTotal * sizeof(float));
        
        memcpy(updateMatrixW1, matrixW1, sizeof(float) * dimensionC * dimension * window);
        memcpy(updateMatrixB1, matrixB1, sizeof(float) * dimensionC);
        memcpy(updateMatrixRelationPr, matrixRelationPr, relationTotal * sizeof(float));				//add
        memcpy(updateMatrixRelation, matrixRelation, dimensionC*relationTotal * sizeof(float));
        //        memcpy(updateFeatureW, featureW, featureLen * sizeof(float));
        //        test::test(0);
        
        
        double maxAvgScore = -100000;
        for (turn = 0; turn < trainTimes; turn ++)
        {
//            printf("previous value\n");
//            for (int i = 0; i < relationTotal; i ++)
//                printf("%lf ", matrixRelationPr[i]);
//            printf("%\n");
//            
//            for (int i = 0; i < PositionTotalE1 * dimensionWPE; i ++)
//                printf("%lf ", positionVecE1[i]);
//            printf("\n");
            nowTurn = turn;
            double rlLoss = 0;
            ok = 0;
            printf("turn = %d\n", turn);
            fprintf(logg, "turn = %d\n", turn);
            fflush(logg);
            printf("alpha = %lf\n", alpha);
            memcpy(positionVecDaoE1, updatePositionVecE1, PositionTotalE1 * dimensionWPE* sizeof(float));
            memcpy(positionVecDaoE2, updatePositionVecE2, PositionTotalE2 * dimensionWPE* sizeof(float));
            memcpy(matrixW1PositionE1Dao, updateMatrixW1PositionE1, dimensionC * dimensionWPE * window* sizeof(float));
            memcpy(matrixW1PositionE2Dao, updateMatrixW1PositionE2, dimensionC * dimensionWPE * window* sizeof(float));
            memcpy(wordVecDao, updateWordVec, dimension * wordTotal * sizeof(float));
            
            memcpy(matrixW1Dao, updateMatrixW1, sizeof(float) * dimensionC * dimension * window);
            memcpy(matrixB1Dao, updateMatrixB1, sizeof(float) * dimensionC);
            memcpy(matrixRelationPrDao, updateMatrixRelationPr, relationTotal * sizeof(float));				//add
            memcpy(matrixRelationDao, updateMatrixRelation, dimensionC*relationTotal * sizeof(float));
            memcpy(featureWDao, featureW, featureLen * sizeof(float));
            
            random_shuffle(shuffleIndex.begin(), shuffleIndex.end());
            //get the vector of sentence and the training loss
            
            score = 0;
            UpdateScore();
            float OneNum = 0;
            double OneScore = 0;
            
            for (map<string,vector<int> >:: iterator it = bags_train.begin(); it!=bags_train.end(); it++)
            {
                if (it->second.size() != 1)
                    continue;
                int i = it->second[0];
                OneScore += lossVec[i];
                OneNum ++;
            }
            //                printf("%lf\n", OneScore / OneNum);
            totAvgScore = OneScore / OneNum;
            
            if (maxAvgScore < (score / trainLists.size()))
            {
                maxAvgScore = score / trainLists.size();
                memcpy(bestPositionVecE1, positionVecE1, PositionTotalE1 * dimensionWPE* sizeof(float));
                memcpy(bestPositionVecE2, positionVecE2, PositionTotalE2 * dimensionWPE* sizeof(float));
                memcpy(bestMatrixW1PositionE1, matrixW1PositionE1, dimensionC * dimensionWPE * window* sizeof(float));
                memcpy(bestMatrixW1PositionE2, matrixW1PositionE2, dimensionC * dimensionWPE * window* sizeof(float));
                memcpy(bestWordVec, wordVec, dimension * wordTotal * sizeof(float));
                
                memcpy(bestMatrixW1, matrixW1, sizeof(float) * dimensionC * dimension * window);
                memcpy(bestMatrixB1, matrixB1, sizeof(float) * dimensionC);
                memcpy(bestMatrixRelationPr, matrixRelationPr, relationTotal * sizeof(float));				//add
                memcpy(bestMatrixRelation, matrixRelation, dimensionC*relationTotal * sizeof(float));
                memcpy(bestFeatureW, featureW, featureLen * sizeof(float));
            }
            printf("finish get sentence vector\n");
            printf("average score = %lf\n", score / trainLists.size());
            fprintf(logg, "average score = %lf\n", score / trainLists.size());
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
                            // if (j == 0)
                            // {
                            //     allChosenSentence.push_back(i);
                            //     rlLoss += lossVec[i];
                            // }
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
                            tmpF += featureList[x] * featureW[x];
                        tmpF = sigmod(tmpF);
                        //                        printf("%lf %lf %d\n", reward[j], avgReward, chosenSenSet[j].size());
                        if (chosenSenSet[j].size() != 0 && i == chosenSenSet[j][l])
                        {
                            // action = 1 ,update gradients
                            for (int x = 0; x < featureList.size(); x ++)
                            {
                                featureWDao[x] += alpha * (reward[j] - avgReward) * (1 - tmpF) * featureList[x];
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
                                featureWDao[x] += alpha * (reward[j] - avgReward) * (-1 * tmpF) * featureList[x];
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
            
//            for (int i = 0; i < featureLen; i ++)
//                printf("%lf ", featureW[i]);
//            printf("\n");
            
            rlLoss /= allChosenSentence.size();
//            if (rlLoss > bestLoss)
//            {
//                bestLoss = rlLoss;
//                if (strcmp(method.c_str(), "rlpre") == 0)
//                    memcpy(bestFeatureW, featureW, featureLen * sizeof(float));
//            }
            
            fprintf(logg, "chosen sentence size = %d %lf %lf\n", allChosenSentence.size(), rlLoss, bestLoss);
            fflush(logg);
            printf("chosen sentence size = %d %lf %lf\n", allChosenSentence.size(), rlLoss, bestLoss);
            
            UpdateCNN();
            
            UpdateValue(&featureW[0], &featureWDao[0], featureLen);
            memcpy(featureWDao, featureW, featureLen * sizeof(float));
//            
//            fclose(fout);
//            outPath = outString + "_chosenInstance.txt" + Int_to_String(turn);
//            fout = fopen(outPath.c_str(), "w");
//            
//            for (int i = 0; i < allChosenSentence.size(); i ++)
//                fprintf(fout, "%d ", allChosenSentence[i]);
//            fprintf(fout, "\n");
//            fclose(fout);
            test::test(1);
            
        }
        
        string tmpPath = outString + "matrixW1+B1.txt";
        FILE *fout = fopen(tmpPath.c_str(), "w");
        //FILE *fout = fopen(("/Users/fengjun/Documents/Research/relation extraction/code/RelationExtraction/out/matrixW1+B1.txt"+version).c_str(), "w");
        fprintf(fout,"%d\t%d\t%d\t%d\n", dimensionC, dimension, window, dimensionWPE);
        for (int i = 0; i < dimensionC; i++) {
            for (int j = 0; j < dimension * window; j++)
                fprintf(fout, "%lf\t",matrixW1[i* dimension*window+j]);
            for (int j = 0; j < dimensionWPE * window; j++)
                fprintf(fout, "%lf\t",matrixW1PositionE1[i* dimensionWPE*window+j]);
            for (int j = 0; j < dimensionWPE * window; j++)
                fprintf(fout, "%lf\t",matrixW1PositionE2[i* dimensionWPE*window+j]);
            fprintf(fout, "%lf\n", matrixB1[i]);
        }
        fclose(fout);
        
        tmpPath = outString + "matrixRl.txt";
        fout = fopen(tmpPath.c_str(), "w");
        fprintf(fout,"%d\t%d\n", relationTotal, dimensionC);
        for (int i = 0; i < relationTotal; i++) {
            for (int j = 0; j < dimensionC; j++)
                fprintf(fout, "%lf\t", bestMatrixRelation[i * dimensionC + j]);
            fprintf(fout, "\n");
        }
        for (int i = 0; i < relationTotal; i++)
            fprintf(fout, "%lf\t", bestMatrixRelationPr[i]);
        fprintf(fout, "\n");
        fclose(fout);
        
        tmpPath = outString + "matrixPosition.txt";
        fout = fopen(tmpPath.c_str(), "w");
        //fout = fopen(("/Users/fengjun/Documents/Research/relation extraction/code/RelationExtraction/out/matrixPosition.txt"+version).c_str(), "w");
        fprintf(fout,"%d\t%d\t%d\n", PositionTotalE1, PositionTotalE2, dimensionWPE);
        for (int i = 0; i < PositionTotalE1; i++) {
            for (int j = 0; j < dimensionWPE; j++)
                fprintf(fout, "%lf\t", bestPositionVecE1[i * dimensionWPE + j]);
            fprintf(fout, "\n");
        }
        for (int i = 0; i < PositionTotalE2; i++) {
            for (int j = 0; j < dimensionWPE; j++)
                fprintf(fout, "%lf\t", bestPositionVecE2[i * dimensionWPE + j]);
            fprintf(fout, "\n");
        }
        fclose(fout);
        
        tmpPath = outString + "word2vec.txt";
        fout = fopen(tmpPath.c_str(), "w");
        //fout = fopen(("/Users/fengjun/Documents/Research/relation extraction/code/RelationExtraction/out/word2vec.txt"+version).c_str(), "w");
        fprintf(fout,"%d\t%d\n",wordTotal,dimension);
        for (int i = 0; i < wordTotal; i++)
        {
            for (int j=0; j<dimension; j++)
                fprintf(fout,"%lf\t", bestWordVec[i*dimension+j]);
            fprintf(fout,"\n");
        }
        fclose(fout);

        string outPath = outString + "bestRL.txt";
        fout = fopen(outPath.c_str(), "w");
        
        for (int i = 0; i < featureLen; i ++)
            fprintf(fout, "%lf ", bestFeatureW[i]);
        fprintf(fout, "\n");
    }
    
    void CountSentenceVec(string pathS)
    {
        sentenceVec.clear();
        lossVec.clear();
        for (int i = 0; i < trainLists.size(); i ++)
        {
            //            if (i > 100) break;
            //                    if (i % 100 == 0) printf("%d\n", i);
            double res = 0;
            double res1 = 0;
            float *r = (float *)calloc(dimensionC, sizeof(float));
            {
                double tmp = GetSentenceVec(0,trainLists[i], trainPositionE1[i], trainPositionE2[i], trainLength[i], headList[i], tailList[i], relationList[i], res, res1, matrixW1, matrixB1, r, matrixRelation, positionVecE1, positionVecE2, matrixW1PositionE1, matrixW1PositionE2, matrixRelationPr, alpha);
                sentenceVec.push_back(r);
                lossVec.push_back(tmp);
            }
            //            free(r);
        }
        
        string scorePath = pathS + "data/pretrain/sentenceVec.txt";
        FILE *fscore = fopen(scorePath.c_str(), "w");
        for (int i = 0; i < sentenceVec.size(); i ++)
        {
            float *r = (float *)calloc(dimensionC, sizeof(float));
            r = sentenceVec[i];
            for (int j = 0; j < dimensionC; j ++)
                fprintf(fscore, "%lf ", r[j]);
            fprintf(fscore, "\n");
        }
        fclose(fscore);
        
        scorePath = pathS + "data/pretrain/lossVec.txt";
        fscore = fopen(scorePath.c_str(), "w");
        for (int i =0; i < lossVec.size(); i ++)
            fprintf(fscore, "%lf\n", lossVec[i]);
        fclose(fscore);
        
    }
    
    void beginTrain()
    {
        string tmpPath = outString + "_log.txt";
        logg = fopen(tmpPath.c_str(), "w");
        
        tmpPath = outString + "_pr.txt";
        prlog = fopen(tmpPath.c_str(), "w");
        
        init();
        preprocess();
        printf("finish preprocess\n");
        selection();
        fclose(logg);
        fclose(prlog);
    }
    
}
