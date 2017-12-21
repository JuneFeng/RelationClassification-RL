//
//  test.h
//  RelationExtraction
//
//  Created by Feng Jun on 06/12/2016.
//  Copyright © 2016 Feng Jun. All rights reserved.
//

#ifndef test_h
#define test_h

#include "init.h"

namespace test{
    
    vector<double> test(int *sentence, int *testPositionE1, int *testPositionE2, int len, float *r);
    
    bool cmp(pair<string, pair<int,double> > a,pair<string, pair<int,double> >b);
    
    void* testMode(void *id );
    
    void test(int flag);
    
    void beginTest(int flag);
    
}
#endif /* test_h */
