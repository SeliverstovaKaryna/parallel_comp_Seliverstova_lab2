#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <cstdlib>
#include <ctime>
#include <iomanip>
 
using namespace std;
using chrono::nanoseconds;
using chrono::duration_cast;
using chrono::high_resolution_clock;
 
void linearExecution(const vector<int>& data, long long& dif, int& maxVal);
void parallelTaskWithMutex(const vector<int>& data, long long& dif, int& maxVal, int numThreads);
void parallelTaskWithCAS(const vector<int>& data, long long& dif, int& maxVal, int numThreads);
 
int main() {
    vector<int> matrixSizes = {10000, 1000000, 100000000, 2000000000};
    vector<int> threadCounts = {8, 16, 32, 64, 128, 256};
 
    cout << "\nTest Results:" << endl;
    cout << "Matrix Size\tThreads\tMode\tTime (seconds)\tDiff\tMax Value" << endl;
 
    for (int matrixSize: matrixSizes) {
        vector<int> data(matrixSize);
        srand(static_cast<unsigned>(time(nullptr)));
        for (int i = 0; i < matrixSize; ++i) {
            data[i] = rand() % 1001;
        }

        long long dif = 0;
        int maxVal = 0;
        auto start = high_resolution_clock::now();
        linearExecution(data, dif, maxVal);
        auto end = high_resolution_clock::now();
        double elapsed = duration_cast<nanoseconds>(end - start).count() * 1e-9;
        cout << matrixSize << "\t\t-\tLinear\t" << fixed << setprecision(6) << elapsed << "\t" << dif << "\t" << maxVal << endl;
 
        cout << endl;
        for (int numThreads: threadCounts) {
            long long dif = 0;
            int maxVal = INT32_MIN;
            auto start = high_resolution_clock::now();
            parallelTaskWithMutex(data, dif, maxVal, numThreads);
            auto end = high_resolution_clock::now();
            double elapsed = duration_cast<nanoseconds>(end - start).count() * 1e-9;
            cout << matrixSize << "\t\t" << numThreads << "\tMutex\t" << fixed << setprecision(6) << elapsed << "\t" <<
                dif << "\t" << maxVal << endl;
        }
        cout << endl;
        for (int numThreads: threadCounts) {
            long long dif = 0;
            int maxVal = INT32_MIN;
            auto start = high_resolution_clock::now();
            parallelTaskWithCAS(data, dif, maxVal, numThreads);
            auto end = high_resolution_clock::now();
            double elapsed = duration_cast<nanoseconds>(end - start).count() * 1e-9;
            cout << matrixSize << "\t\t" << numThreads << "\tCAS\t" << fixed << setprecision(6) << elapsed << "\t" <<
                dif << "\t" << maxVal << endl;
        }
        cout << endl << endl;
    }
 
    return 0;
}
 
void linearExecution(const vector<int> &data, long long &dif, int & maxVal) {
    dif = 0;
    maxVal = INT32_MIN;
    for (int value: data) {
        if (value % 2 == 0) {
            if (dif == 0) {
                dif = value;
            }
            else {
                dif -= value;
            }
            if (value > maxVal) {
                maxVal = value;
            }
        }
    }
}
 
void taskWithMutex(int start, int end, const vector<int>& data, long long& localDif, int& localMax, mutex& mtx) {
    long long dif = 0;
    int maxVal = INT32_MIN;
    for (int i = start; i < end; ++i) {
        if (data[i] % 2 == 0) {
            if (dif == 0) {
                dif = data[i];
            }
            else {
                dif -= data[i];
            }
            if (data[i] > maxVal) {
                maxVal = data[i];
            }
        }
    }
    lock_guard<mutex> lock(mtx);
    localDif += dif;

    if (maxVal > localMax) {
        localMax = maxVal;
    }
}

void parallelTaskWithMutex(const vector<int> &data, long long &dif, int & maxVal, int numThreads) {
    dif = 0;
    maxVal = INT32_MIN;

    mutex mtx;
    vector<thread> threads;
 
    int threadsPerSection = data.size() / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int start = i * threadsPerSection;
        int end = (i == numThreads - 1) ? data.size() : start + threadsPerSection;
        threads.emplace_back(taskWithMutex, start, end, cref(data), ref(dif), ref(maxVal), ref(mtx));
    }
 
    for (auto &th: threads) {
        if (th.joinable()) {
            th.join();
        }
    }
}
 
void taskWithCAS(int start, int end, const vector<int> &data, atomic<long long> &atomicDif, atomic<int> &atomicMax, atomic<long long>& atomicDifFirstVal) {
    long long localDif = 0;
    int localMax = INT32_MIN;
    bool isFirstInSection = true;

    for (int i = start; i < end; ++i) {
            if (data[i] % 2 == 0) {
                if (isFirstInSection) {
                    localDif = atomicDifFirstVal.load(memory_order_relaxed);
                    localDif -= data[i];
                    isFirstInSection = false;
                }
                else {
                    localDif -= data[i];
                }
                if (data[i] > localMax) {
                    localMax = data[i];
                }
            }
    }
 
    long long currentDif = atomicDif.load(memory_order_relaxed);
    long long newDif = currentDif + localDif;

    while (atomicDif.compare_exchange_weak(currentDif, newDif, memory_order_relaxed)) {
        newDif = currentDif + localDif;
    }
 
    int currentMax = atomicMax.load(memory_order_relaxed);
    while (localMax > currentMax && atomicMax.compare_exchange_weak(currentMax, localMax, memory_order_relaxed)) {
    }
}
 
void parallelTaskWithCAS(const vector<int> &data, long long &dif, int &maxVal, int numThreads) {
    atomic<long long> atomicDif(0);
    atomic<int> atomicMax(INT32_MIN);
    vector<thread> threads;
    atomic<long long> atomicDifFirstVal(0);

    for (int value : data) {
        if (value % 2 == 0) {
            atomicDifFirstVal = value;
            break;
        }
    }

    int threadsPerSection = data.size() / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int start = i * threadsPerSection;
        int end = (i == numThreads - 1) ? data.size() : start + threadsPerSection;
        threads.emplace_back(taskWithCAS, start, end, cref(data), ref(atomicDif), ref(atomicMax), ref(atomicDifFirstVal));
    }
 
    for (auto &th: threads) {
        if (th.joinable()) {
            th.join();
        }
    }
 
    dif = atomicDif.load();
    maxVal = atomicMax.load();
}
