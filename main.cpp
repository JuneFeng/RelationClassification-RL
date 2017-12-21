//
//  main.cpp
//  RelationExtraction
//
//  Created by Feng Jun on 06/12/2016.
//  Copyright © 2016 Feng Jun. All rights reserved.
//

#include <iostream>
#include "test.h"
#include "RL.h"
#include "RLPre.h"


double GetDoubleNum(string s)
{
    double tmp = 0;
    int startIndex = -1;
    for (int i = 0; i < s.size(); i ++)
    {
        if (s[i] == '.')
        {
            startIndex = i + 1;
            break;
        }
        tmp = tmp * 10 + s[i] - '0';
    }
    double base = 0.1;
    for (int i = startIndex; i < s.size(); i ++)
    {
        tmp += (s[i] - '0') * base;
        base /= 10;
    }
    
    return tmp;
}


//vector<double *> testSentence;

int main(int argc, const char * argv[]) {

    //    srand( (unsigned)time( NULL ) );
//    if (strcmp(argv[2], "server") == 0)
//        pathString = serverPathString;
//    else pathString = localPathString;
    pathString = serverPathString;
    outString = pathString + "out/" + argv[1] + "/";
    //    outString = pathString + "/data/pretrain/orignalcnn";
    if (strcmp(argv[1], "test") == 0)
    {
        outString = pathString + "out/rl/";
        test::beginTest(0);
        return 0;
    }
    method = argv[1];
    InitialAlpha = GetDoubleNum(argv[2]);
    if (strcmp(argv[1], "rl") == 0)
        RL::beginTrain();
    if (strcmp(argv[1], "rlpre") == 0)
        RLPre::beginTrain();
    if (strcmp(argv[1], "countvec") == 0)
    {
        RL::CountSentenceVec(pathString);
    }
    
    return 0;
}
