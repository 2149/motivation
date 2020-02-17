
#include <iostream>
#include <chrono>
#include <random>
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include "tbb/tbb.h"

using namespace std;

#include "../P-ART/Tree.h"
#include "../third-party/FAST_FAIR/btree.h"
#include "../third-party/CCEH/src/Level_hashing.h"
#include "../third-party/CCEH/src/CCEH.h"
#include "../third-party/WOART/woart.h"
#include "../P-BwTree/src/bwtree.h"
#include "masstree.h"
#include "clht.h"
#include "ssmem.h"

#ifdef HOT
#include <hot/rowex/HOTRowex.hpp>
#include <idx/contenthelpers/IdentityKeyExtractor.hpp>
#include <idx/contenthelpers/OptionalValue.hpp>
#endif

#include "test_common.h"
#include "statistic.h"

using namespace wangziqi2013::bwtree;

// index types
enum {
    TYPE_ART,
    TYPE_HOT,
    TYPE_BWTREE,
    TYPE_MASSTREE,
    TYPE_CLHT,
    TYPE_FASTFAIR,
    TYPE_LEVELHASH,
    TYPE_CCEH,
    TYPE_WOART,
};

enum {
    OP_INSERT,
    OP_READ,
    OP_SCAN,
    OP_DELETE,
};

enum {
    WORKLOAD_A,
    WORKLOAD_B,
    WORKLOAD_C,
    WORKLOAD_D,
    WORKLOAD_E,
};

enum {
    RANDINT_KEY,
    STRING_KEY,
};

enum {
    UNIFORM,
    ZIPFIAN,
};

namespace Dummy {
    inline void mfence() {asm volatile("mfence":::"memory");}

    inline void clflush(char *data, int len, bool front, bool back)
    {
        if (front)
            mfence();
        volatile char *ptr = (char *)((unsigned long)data & ~(64 - 1));
        for (; ptr < data+len; ptr += 64){
#ifdef CLFLUSH
            asm volatile("clflush %0" : "+m" (*(volatile char *)ptr));
#elif CLFLUSH_OPT
            asm volatile(".byte 0x66; clflush %0" : "+m" (*(volatile char *)(ptr)));
#elif CLWB
            asm volatile(".byte 0x66; xsaveopt %0" : "+m" (*(volatile char *)(ptr)));
#endif
        }
        if (back)
            mfence();
    }
}


////////////////////////Helper functions for P-BwTree/////////////////////////////
/*
 * class KeyComparator - Test whether BwTree supports context
 *                       sensitive key comparator
 *
 * If a context-sensitive KeyComparator object is being used
 * then it should follow rules like:
 *   1. There could be no default constructor
 *   2. There MUST be a copy constructor
 *   3. operator() must be const
 *
 */
class KeyComparator {
 public:
  inline bool operator()(const long int k1, const long int k2) const {
    return k1 < k2;
  }

  inline bool operator()(const uint64_t k1, const uint64_t k2) const {
      return k1 < k2;
  }

  inline bool operator()(const char *k1, const char *k2) const {
      return memcmp(k1, k2, strlen(k1) > strlen(k2) ? strlen(k1) : strlen(k2)) < 0;
  }

  KeyComparator(int dummy) {
    (void)dummy;

    return;
  }

  KeyComparator() = delete;
  //KeyComparator(const KeyComparator &p_key_cmp_obj) = delete;
};

/*
 * class KeyEqualityChecker - Tests context sensitive key equality
 *                            checker inside BwTree
 *
 * NOTE: This class is only used in KeyEqual() function, and is not
 * used as STL template argument, it is not necessary to provide
 * the object everytime a container is initialized
 */
class KeyEqualityChecker {
 public:
  inline bool operator()(const long int k1, const long int k2) const {
    return k1 == k2;
  }

  inline bool operator()(uint64_t k1, uint64_t k2) const {
      return k1 == k2;
  }

  inline bool operator()(const char *k1, const char *k2) const {
      if (strlen(k1) != strlen(k2))
          return false;
      else
          return memcmp(k1, k2, strlen(k1)) == 0;
  }

  KeyEqualityChecker(int dummy) {
    (void)dummy;

    return;
  }

  KeyEqualityChecker() = delete;
  //KeyEqualityChecker(const KeyEqualityChecker &p_key_eq_obj) = delete;
};
/////////////////////////////////////////////////////////////////////////////////

////////////////////////Helper functions for P-HOT/////////////////////////////
typedef struct IntKeyVal {
    uint64_t key;
    uintptr_t value;
} IntKeyVal;

template<typename ValueType = IntKeyVal *>
class IntKeyExtractor {
    public:
    typedef uint64_t KeyType;

    inline KeyType operator()(ValueType const &value) const {
        return value->key;
    }
};

template<typename ValueType = Key *>
class KeyExtractor {
    public:
    typedef char const * KeyType;

    inline KeyType operator()(ValueType const &value) const {
        return (char const *)value->fkey;
    }
};
/////////////////////////////////////////////////////////////////////////////////

////////////////////////Helper functions for P-CLHT/////////////////////////////
typedef struct thread_data {
    uint32_t id;
    clht_t *ht;
} thread_data_t;

typedef struct barrier {
    pthread_cond_t complete;
    pthread_mutex_t mutex;
    int count;
    int crossing;
} barrier_t;

void barrier_init(barrier_t *b, int n) {
    pthread_cond_init(&b->complete, NULL);
    pthread_mutex_init(&b->mutex, NULL);
    b->count = n;
    b->crossing = 0;
}

void barrier_cross(barrier_t *b) {
    pthread_mutex_lock(&b->mutex);
    b->crossing++;
    if (b->crossing < b->count) {
        pthread_cond_wait(&b->complete, &b->mutex);
    } else {
        pthread_cond_broadcast(&b->complete);
        b->crossing = 0;
    }
    pthread_mutex_unlock(&b->mutex);
}

barrier_t barrier;
/////////////////////////////////////////////////////////////////////////////////

static uint64_t LOAD_SIZE = 4000000;
static uint64_t RUN_SIZE = 1000000;

void loadKey(TID tid, Key &key) {
    return ;
}

#define MAX_THREAD 128
static int rand_seed[MAX_THREAD];

static int SetRandSeed(void) {
    for(int i =0; i < MAX_THREAD; i ++) {
        rand_seed[i] = 0xdeadbeef * (i + 1);
    }
}

void motivation_run_randint(int index_type, int wl, int kt, int ap, int num_thread,
        std::vector<uint64_t> &init_keys,
        std::vector<uint64_t> &keys,
        std::vector<int> &ranges,
        std::vector<int> &ops)
{
    std::string op;
    uint64_t key;
    int range;
    Statistic stats;


    rocksdb::Random64 *rnd_insert[num_thread];
    rocksdb::Random64 *rnd_get[num_thread];
    rocksdb::Random64 *rnd_scan[num_thread];
    rocksdb::Random64 *rnd_delete[num_threadn];

    std::atomic<int> range_complete, range_incomplete;
    range_complete.store(0);
    range_incomplete.store(0);

    {
        for(int i = 0; i < num_thread; i ++) {
            rnd_insert[i] = new rocksdb::Random64(0xdeadbeef * (i + 1)); 
            rnd_get[i] = new rocksdb::Random64(0xdeadbeef * (i + 1)); 
            rnd_scan[i] = new rocksdb::Random64(0xdeadbeef * (i + 1)); 
            rnd_delete[i] = new rocksdb::Random64(0xdeadbeef * (i + 1)); 
        }
    }
    std::atomic<int> next_thread_id;
    if (index_type == TYPE_ART) {
        printf("Motivation test ART start!\n");
        ART_ROWEX::Tree tree(loadKey);
        {
            // Load
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            tbb::parallel_for(tbb::blocked_range<uint64_t>(0, LOAD_SIZE), [&](const tbb::blocked_range<uint64_t> &scope) {
                auto t = tree.getThreadInfo();
                int thread_id = next_thread_id.fetch_add(1);
                for (uint64_t i = scope.begin(); i != scope.end(); i++) {
                    uint64_t key_64 = rnd_insert[thread_id]->Next();
                    stats.start();
                    Key *key = key->make_leaf(key_64, sizeof(uint64_t), key_64);
                    tree.insert(key, t);
                    stats.end();
                    stats.add_put();
                    if ((i % 1000) == 0) {
                        stats.PrintLatency(i);
                    }
                }
            });
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - starttime);
            printf("Load_Throughput: load, %f ,ops/s\n", (LOAD_SIZE * 1.0) / duration.count() * 1000000);
        }

        {
            // Put
            Key *end = end->make_leaf(UINT64_MAX, sizeof(uint64_t), 0);
            next_thread_id.store(0);
            next_thread_id.store(0);std::chrono::system_clock::now();
            tbb::parallel_for(tbb::blocked_range<uint64_t>(0, RUN_SIZE), [&](const tbb::blocked_range<uint64_t> &scope) {
                auto t = tree.getThreadInfo();
                int thread_id = next_thread_id.fetch_add(1);
                for (uint64_t i = scope.begin(); i != scope.end(); i++) {
                    uint64_t key_64 = rnd_insert[thread_id]->Next();
                    Key *key = key->make_leaf(key_64, sizeof(uint64_t), key_64);
                    tree.insert(key, t);
                }
            });
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - starttime);
            printf("Put_Throughput: run, %f ,ops/s\n", (RUN_SIZE * 1.0) / duration.count() * 1000000);
        }

        {
            // Get
            Key *end = end->make_leaf(UINT64_MAX, sizeof(uint64_t), 0);
            next_thread_id.store(0);std::chrono::system_clock::now();
            tbb::parallel_for(tbb::blocked_range<uint64_t>(0, RUN_SIZE), [&](const tbb::blocked_range<uint64_t> &scope) {
                int thread_id = next_thread_id.fetch_add(1);
                auto t = tree.getThreadInfo();
                for (uint64_t i = scope.begin(); i != scope.end(); i++) {
                    uint64_t key_64 = rnd_get[thread_id]->Next();
                    Key *key = key->make_leaf(key_64, sizeof(uint64_t), 0);
                    uint64_t *val = reinterpret_cast<uint64_t *>(tree.lookup(key, t));
                    if (*val != key_64) {
                        std::cout << "[ART] wrong key read: " << val << " expected:" << key_64 << std::endl;
                        exit(1);
                    }
                }
            });
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - starttime);
            printf("Get_Throughput: run, %f ,ops/s\n", (RUN_SIZE * 1.0) / duration.count() * 1000000);
        }

        {
            // Scan
            Key *end = end->make_leaf(UINT64_MAX, sizeof(uint64_t), 0);
            int count = 4;
            int scan_times = 100;
            int scan_count = 100;
            while(count >0 ) {
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            tbb::parallel_for(tbb::blocked_range<uint64_t>(0, scan_times), [&](const tbb::blocked_range<uint64_t> &scope) {
                auto t = tree.getThreadInfo();
                int thread_id = next_thread_id.fetch_add(1);
                for (uint64_t i = scope.begin(); i != scope.end(); i++) {
                    uint64_t key_64 = rnd_scan[thread_id]->Next();
                    Key *results[scan_count];
                    Key *continueKey = NULL;
                    size_t resultsFound = 0;
                    size_t resultsSize = scan_count;
                    Key *start = start->make_leaf(key_64, sizeof(uint64_t), 0);
                    tree.lookupRange(start, end, continueKey, results, resultsSize, resultsFound, t);
                }
            });
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - starttime);
            printf("Scan_%d_Throughput: run, %f ,ops/s\n", scan_count, (scan_times * 1.0) / duration.count() * 1000000);
            scan_count *= 10;
            count --;
            }
        }

        {
            // Delete
            Key *end = end->make_leaf(UINT64_MAX, sizeof(uint64_t), 0);
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            tbb::parallel_for(tbb::blocked_range<uint64_t>(0, RUN_SIZE), [&](const tbb::blocked_range<uint64_t> &scope) {
                auto t = tree.getThreadInfo();
                int thread_id = next_thread_id.fetch_add(1);
                for (uint64_t i = scope.begin(); i != scope.end(); i++) {
                    uint64_t key_64 = rnd_delete[next_thread_id]->Next();
                    Key *key = key->make_leaf(key_64, sizeof(uint64_t), 0);
                    tree.remove(key, t);
                }
            });
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - starttime);
            printf("Delete_Throughput: run, %f ,ops/s\n", (RUN_SIZE * 1.0) / duration.count() * 1000000);
        }
#ifdef HOT
    } else if (index_type == TYPE_HOT) {
        // printf("No implie for HOT with motivation!\n");
        printf("Motivation test HOT start!\n");
        hot::rowex::HOTRowex<IntKeyVal *, IntKeyExtractor> mTrie;
        {
            // Load
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            tbb::parallel_for(tbb::blocked_range<uint64_t>(0, LOAD_SIZE), [&](const tbb::blocked_range<uint64_t> &scope) {
                int thread_id = next_thread_id.fetch_add(1);
                for (uint64_t i = scope.begin(); i != scope.end(); i++) {
                    IntKeyVal *key;
                    uint64_t key_64 = rnd_insert[thread_id]->Next();
                    stats.start();
                    posix_memalign((void **)&key, 64, sizeof(IntKeyVal));
                    key->key = key_64; key->value = key_64;
                    Dummy::clflush((char *)key, sizeof(IntKeyVal), true, true);
                    if (!(mTrie.insert(key))) {
                        fprintf(stderr, "[HOT] load insert fail\n");
                        exit(1);
                    }
                    stats.end();
                    stats.add_put();
                    if ((i % 1000) == 0) {
                        stats.PrintLatency(i);
                    }
                }
            });
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - starttime);
            printf("Load_Throughput: load, %f ,ops/s\n", (LOAD_SIZE * 1.0) / duration.count() * 1000000);
        }

        {
            // Put
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            tbb::parallel_for(tbb::blocked_range<uint64_t>(0, RUN_SIZE), [&](const tbb::blocked_range<uint64_t> &scope) {
                int thread_id = next_thread_id.fetch_add(1);
                for (uint64_t i = scope.begin(); i != scope.end(); i++) {
                    IntKeyVal *key;
                    uint64_t key_64 = rnd_insert[thread_id]->Next();
                    posix_memalign((void **)&key, 64, sizeof(IntKeyVal));
                    key->key = key_64; key->value = key_64;
                    Dummy::clflush((char *)key, sizeof(IntKeyVal), true, true);
                    if (!(mTrie.insert(key))) {
                        fprintf(stderr, "[HOT] run insert fail\n");
                        exit(1);
                    }
                }
            });
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - starttime);
            printf("Put_Throughput: run, %f ,ops/s\n", (RUN_SIZE * 1.0) / duration.count() * 1000000);
        }

        {
            // Get
            int notfound = 0;
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            tbb::parallel_for(tbb::blocked_range<uint64_t>(0, RUN_SIZE), [&](const tbb::blocked_range<uint64_t> &scope) {
                int thread_id = next_thread_id.fetch_add(1);
                for (uint64_t i = scope.begin(); i != scope.end(); i++) {
                    uint64_t key_64 = rnd_get[thread_id]->Next();
                    idx::contenthelpers::OptionalValue<IntKeyVal *> result = mTrie.lookup(key_64);
                    if (!result.mIsValid || result.mValue->value != key_64) {
                        notfound ++;
                        // printf("mIsValid = %d\n", result.mIsValid);
                        // printf("Return value = %lu, Correct value = %lu\n", result.mValue->value, key_64);
                        // exit(1);
                    }
                }
            });
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - starttime);
            printf("Get_Throughput: run, %f ,ops/s\n", (RUN_SIZE * 1.0) / duration.count() * 1000000);
            printf("Not found key %d\n", notfound);
        }

        {
            // Scan
            int count = 4;
            int scan_times = 100;
            int scan_count = 100;
            
            while(count > 0) {
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            tbb::parallel_for(tbb::blocked_range<uint64_t>(0, scan_times), [&](const tbb::blocked_range<uint64_t> &scope) {
                int thread_id = next_thread_id.fetch_add(1);
                for (uint64_t i = scope.begin(); i != scope.end(); i++) {
                    uint64_t key_64 = rnd_scan[thread_id]->Next();
                    uintptr_t buf[scan_count];
                    hot::rowex::HOTRowexSynchronizedIterator<IntKeyVal *, IntKeyExtractor> it = mTrie.lower_bound(key_64);
                    int resultsFound = 0;
                    while (it != mTrie.end() && resultsFound < scan_count) {
                        buf[resultsFound] = (*it)->value;
                        resultsFound++;
                        ++it;
                    }
                    printf("Found %d, while scan %d\n", resultsFound, scan_count);
                    // idx::contenthelpers::OptionalValue<IntKeyVal *> result = mTrie.scan(key_64, scan_count);
                }
            });
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - starttime);
            printf("Scan_%d_Throughput: run, %f ,ops/s\n", scan_count, (scan_times * 1.0) / duration.count() * 1000000);
            scan_count *= 10;
            count --;
            }
        }

        // {
        //     // Delete
        //     auto starttime = std::chrono::system_clock::now();
        //     tbb::parallel_for(tbb::blocked_range<uint64_t>(0, 10000), [&](const tbb::blocked_range<uint64_t> &scope) {
        //         for (uint64_t i = scope.begin(); i != scope.end(); i++) {
        //             uint64_t key_64 = rnd_scan[thread_id]->Next();
        //             idx::contenthelpers::OptionalValue<IntKeyVal *> result = mTrie.scan(key_64, 100);
        //         }
        //     });
        //     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        //             std::chrono::system_clock::now() - starttime);
        //     printf("Scan_Throughput: run, %f ,ops/s\n", (10000 * 1.0) / duration.count() * 1000000);
        // }

#endif
    } else if (index_type == TYPE_BWTREE) {
        // printf("No implie for BWTREE with motivation!\n");
        printf("Motivation test BWTREE start!\n");
        auto t = new BwTree<uint64_t, uint64_t, KeyComparator, KeyEqualityChecker>{true, KeyComparator{1}, KeyEqualityChecker{1}};
        t->UpdateThreadLocal(1);
        t->AssignGCID(0);
        std::atomic<int> next_thread_id;

        {
            // Load
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            t->UpdateThreadLocal(num_thread);
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = LOAD_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + LOAD_SIZE / num_thread;

                t->AssignGCID(thread_id);
                for (uint64_t i = start_key; i < end_key; i++) {
                    uint64_t key_64 = rnd_insert[thread_id]->Next();
                    stats.start();
                    t->Insert(key_64, key_64);
                    stats.end();
                    stats.add_put();
                    if ((i % 1000) == 0) {
                        stats.PrintLatency(i);
                    }
                }
                t->UnregisterThread(thread_id);
            };

            std::vector<std::thread> thread_group;

            for (int i = 0; i < num_thread; i++)
                thread_group.push_back(std::thread{func});

            for (int i = 0; i < num_thread; i++)
                thread_group[i].join();
            t->UpdateThreadLocal(1);
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - starttime);
            printf("Load_Throughput: load, %f ,ops/s\n", (LOAD_SIZE * 1.0) / duration.count() * 1000000);
        }

        {
            // Put
            auto starttime = std::chrono::system_clock::now();
            next_thread_id.store(0);
            t->UpdateThreadLocal(num_thread);
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = RUN_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + RUN_SIZE / num_thread;

                t->AssignGCID(thread_id);
                for (uint64_t i = start_key; i < end_key; i++) {
                    uint64_t key_64 = rnd_insert[thread_id]->Next();
                    t->Insert(key_64, key_64);
                }
                t->UnregisterThread(thread_id);
            };

            std::vector<std::thread> thread_group;

            for (int i = 0; i < num_thread; i++)
                thread_group.push_back(std::thread{func});

            for (int i = 0; i < num_thread; i++)
                thread_group[i].join();
            t->UpdateThreadLocal(1);
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - starttime);
            printf("Put_Throughput: run, %f ,ops/s\n", (RUN_SIZE * 1.0) / duration.count() * 1000000);
        }

        {
            // Get
            auto starttime = std::chrono::system_clock::now();
            next_thread_id.store(0);
            t->UpdateThreadLocal(num_thread);
            auto func = [&]() {
                std::vector<uint64_t> v{};
                v.reserve(1);
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = RUN_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + RUN_SIZE / num_thread;

                t->AssignGCID(thread_id);
                for (uint64_t i = start_key; i < end_key; i++) {
                    uint64_t key_64 = rnd_get[thread_id]->Next();
                    v.clear();
                    t->GetValue(key_64, v);
                    if (v[0] != key_64) {
                        std::cout << "[BWTREE] wrong key read: " << v[0] << " expected:" << key_64 << std::endl;
                    }  
                } 
            };

            std::vector<std::thread> thread_group;

            for (int i = 0; i < num_thread; i++)
                thread_group.push_back(std::thread{func});

            for (int i = 0; i < num_thread; i++)
                thread_group[i].join();
            t->UpdateThreadLocal(1);
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - starttime);
            printf("Get_Throughput: run, %f ,ops/s\n", (RUN_SIZE * 1.0) / duration.count() * 1000000);
        }

        {
            // Scan
            int count = 4;
            int scan_times = 100;
            int scan_count = 100;
            while(count >0 ) {
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            t->UpdateThreadLocal(num_thread);
            auto func = [&]() {
                std::vector<uint64_t> v{};
                v.reserve(1);
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = scan_times / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + scan_times / num_thread;

                t->AssignGCID(thread_id);
                for (uint64_t i = start_key; i < end_key; i++) {
                    uint64_t buf[scan_count];
                    uint64_t key_64 = rnd_scan[thread_id]->Next();
                    auto it = t->Begin(key_64);

                    int resultsFound = 0;
                    while (it.IsEnd() != true && resultsFound != scan_count) {
                        buf[resultsFound] = it->second;
                        resultsFound++;
                        it++;
                    }
                }
            };

            std::vector<std::thread> thread_group;

            for (int i = 0; i < num_thread; i++)
                thread_group.push_back(std::thread{func});

            for (int i = 0; i < num_thread; i++)
                thread_group[i].join();
            t->UpdateThreadLocal(1);
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - starttime);
            printf("Scan_%d_Throughput: run, %f ,ops/s\n", scan_count, (scan_times * 1.0) / duration.count() * 1000000);
            scan_count *= 10;
            count --;
            }
        }
        {
            // Delete
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            t->UpdateThreadLocal(num_thread);
            auto func = [&]() {
                std::vector<uint64_t> v{};
                v.reserve(1);
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = RUN_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + RUN_SIZE / num_thread;

                t->AssignGCID(thread_id);
                for (uint64_t i = start_key; i < end_key; i++) {
                    uint64_t key_64 = rnd_delete[next_thread_id]->Next();
                    t->Delete(key_64, key_64);
                }
            };

            std::vector<std::thread> thread_group;

            for (int i = 0; i < num_thread; i++)
                thread_group.push_back(std::thread{func});

            for (int i = 0; i < num_thread; i++)
                thread_group[i].join();
            t->UpdateThreadLocal(1);
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - starttime);
            printf("Delete_Throughput: run, %f ,ops/s\n", (RUN_SIZE * 1.0) / duration.count() * 1000000);
        }

    } else if (index_type == TYPE_MASSTREE) {
        // printf("No implie for MASSTREE with motivation!\n");
        printf("Motivation test MASSTREE start!\n");
        masstree::leafnode *init_root = new masstree::leafnode(0);
        masstree::masstree *tree = new masstree::masstree(init_root);

        {
            // Load
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            tbb::parallel_for(tbb::blocked_range<uint64_t>(0, LOAD_SIZE), [&](const tbb::blocked_range<uint64_t> &scope) {
                int thread_id = next_thread_id.fetch_add(1);
                for (uint64_t i = scope.begin(); i != scope.end(); i++) {
                    uint64_t key_64 = rnd_insert[thread_id]->Next();
                    stats.start();
                    tree->put(key_64, (void *)key_64);
                    stats.end();
                    stats.add_put();
                    if ((i % 1000) == 0) {
                        stats.PrintLatency(i);
                    }
                }
            });
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - starttime);
            printf("Load_Throughput: load, %f ,ops/s\n", (LOAD_SIZE * 1.0) / duration.count() * 1000000);
        }

        {
            // Put
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            tbb::parallel_for(tbb::blocked_range<uint64_t>(0, RUN_SIZE), [&](const tbb::blocked_range<uint64_t> &scope) {
                int thread_id = next_thread_id.fetch_add(1);
                for (uint64_t i = scope.begin(); i != scope.end(); i++) {
                    uint64_t key_64 = rnd_insert[thread_id]->Next();
                    tree->put(key_64, (void *)key_64);
                }
            });
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - starttime);
            printf("Put_Throughput: run, %f ,ops/s\n", (RUN_SIZE * 1.0) / duration.count() * 1000000);
        }

        {
            // Get
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            tbb::parallel_for(tbb::blocked_range<uint64_t>(0, RUN_SIZE), [&](const tbb::blocked_range<uint64_t> &scope) {
                int thread_id = next_thread_id.fetch_add(1);
                for (uint64_t i = scope.begin(); i != scope.end(); i++) {
                    uint64_t key_64 = rnd_get[thread_id]->Next();
                    uint64_t *ret = reinterpret_cast<uint64_t *> (tree->get(key_64));
                    if ((uint64_t)ret != key_64) {
                        printf("[MASS] search key = %lu, search value = %lu\n", key_64, (uint64_t)ret);
                        exit(1);
                    }
                }
            });
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - starttime);
            printf("Get_Throughput: run, %f ,ops/s\n", (RUN_SIZE * 1.0) / duration.count() * 1000000);
        }

        {
            // Scan
            Key *end = end->make_leaf(UINT64_MAX, sizeof(uint64_t), 0);
            int count = 4;
            int scan_times = 100;
            int scan_count = 100;
            while(count >0 ) {
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            tbb::parallel_for(tbb::blocked_range<uint64_t>(0, 10000), [&](const tbb::blocked_range<uint64_t> &scope) {
                int thread_id = next_thread_id.fetch_add(1);
                for (uint64_t i = scope.begin(); i != scope.end(); i++) {
                    uint64_t key_64 = rnd_scan[thread_id]->Next();
                    uint64_t buf[scan_count];
                    int ret = tree->scan(key_64, scan_count, buf);
                }
            });
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - starttime);
            printf("Scan_%d_Throughput: run, %f ,ops/s\n", scan_count, (scan_times * 1.0) / duration.count() * 1000000);
            scan_count *= 10;
            count --;
            }
        }

        {
            // Delete
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            tbb::parallel_for(tbb::blocked_range<uint64_t>(0, RUN_SIZE), [&](const tbb::blocked_range<uint64_t> &scope) {
                int thread_id = next_thread_id.fetch_add(1);
                for (uint64_t i = scope.begin(); i != scope.end(); i++) {
                    uint64_t key_64 = rnd_delete[next_thread_id]->Next();
                    tree->del(key_64);
                }
            });
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - starttime);
            printf("Delete_Throughput: run, %f ,ops/s\n", (RUN_SIZE * 1.0) / duration.count() * 1000000);
        }

    } else if (index_type == TYPE_CLHT) {
        // printf("No implie for CLHT with motivation!\n");
        printf("Motivation test CLHT start!\n");

        clht_t *hashtable = clht_create(512);

        barrier_init(&barrier, num_thread);

        thread_data_t *tds = (thread_data_t *) malloc(num_thread * sizeof(thread_data_t));

        {
            // Load
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                tds[thread_id].id = thread_id;
                tds[thread_id].ht = hashtable;

                uint64_t start_key = LOAD_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + LOAD_SIZE / num_thread;

                clht_gc_thread_init(tds[thread_id].ht, tds[thread_id].id);
                barrier_cross(&barrier);

                for (uint64_t i = start_key; i < end_key; i++) {
                    uint64_t key_64 = rnd_insert[thread_id]->Next();
                    stats.start();
                    clht_put(tds[thread_id].ht, key_64, key_64);
                    stats.end();
                    stats.add_put();
                    if ((i % 1000) == 0) {
                        stats.PrintLatency(i);
                    }
                }
            };

            std::vector<std::thread> thread_group;

            for (int i = 0; i < num_thread; i++)
                thread_group.push_back(std::thread{func});

            for (int i = 0; i < num_thread; i++)
                thread_group[i].join();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - starttime);
            printf("Load_Throughput: load, %f ,ops/s\n", (LOAD_SIZE * 1.0) / duration.count() * 1000000);
        }

        barrier.crossing = 0;

        {
            // Put
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                tds[thread_id].id = thread_id;
                tds[thread_id].ht = hashtable;

                uint64_t start_key = RUN_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + RUN_SIZE / num_thread;

                clht_gc_thread_init(tds[thread_id].ht, tds[thread_id].id);
                barrier_cross(&barrier);

                for (uint64_t i = start_key; i < end_key; i++) {
                    uint64_t key_64 = rnd_insert[thread_id]->Next();
                    clht_put(tds[thread_id].ht, key_64, key_64);
                }
            };

            std::vector<std::thread> thread_group;

            for (int i = 0; i < num_thread; i++)
                thread_group.push_back(std::thread{func});

            for (int i = 0; i < num_thread; i++)
                thread_group[i].join();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - starttime);
            printf("Put_Throughput: run, %f ,ops/s\n", (RUN_SIZE * 1.0) / duration.count() * 1000000);
        }
        barrier.crossing = 0;
        {
            // Get
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                tds[thread_id].id = thread_id;
                tds[thread_id].ht = hashtable;

                uint64_t start_key = RUN_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + RUN_SIZE / num_thread;

                clht_gc_thread_init(tds[thread_id].ht, tds[thread_id].id);
                barrier_cross(&barrier);

                for (uint64_t i = start_key; i < end_key; i++) {
                    uint64_t key_64 = rnd_get[thread_id]->Next();
                    uintptr_t val = clht_get(tds[thread_id].ht->ht, key_64);
                    if (val != key_64) {
                        std::cout << "[CLHT] wrong key read: " << val << "expected: " << key_64 << std::endl;
                        exit(1);
                    }
                }
            };

            std::vector<std::thread> thread_group;

            for (int i = 0; i < num_thread; i++)
                thread_group.push_back(std::thread{func});

            for (int i = 0; i < num_thread; i++)
                thread_group[i].join();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - starttime);
            printf("Get_Throughput: run, %f ,ops/s\n", (RUN_SIZE * 1.0) / duration.count() * 1000000);
        }

        barrier.crossing = 0;
        {
            // Delete
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                tds[thread_id].id = thread_id;
                tds[thread_id].ht = hashtable;

                uint64_t start_key = RUN_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + RUN_SIZE / num_thread;

                clht_gc_thread_init(tds[thread_id].ht, tds[thread_id].id);
                barrier_cross(&barrier);

                for (uint64_t i = start_key; i < end_key; i++) {
                    uint64_t key_64 = rnd_insert[thread_id]->Next();
                    auto ret = clht_remove(tds[thread_id].ht, key_64);
                }
            };

            std::vector<std::thread> thread_group;

            for (int i = 0; i < num_thread; i++)
                thread_group.push_back(std::thread{func});

            for (int i = 0; i < num_thread; i++)
                thread_group[i].join();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - starttime);
            printf("Delete_Throughput: run, %f ,ops/s\n", (RUN_SIZE * 1.0) / duration.count() * 1000000);
        }
        clht_gc_destroy(hashtable);
    } else if (index_type == TYPE_FASTFAIR) {
        printf("No implie for MASSTREE with motivation!\n");
    } else if (index_type == TYPE_LEVELHASH) {
        printf("No implie for MASSTREE with motivation!\n");
    } else if (index_type == TYPE_CCEH) {
        printf("No implie for MASSTREE with motivation!\n");
    } else if (index_type == TYPE_WOART) {
        printf("No implie for MASSTREE with motivation!\n");
#ifndef STRING_TYPE
        
#endif
    }

    {
        for(int i = 0; i < num_thread; i ++) {
            delete rnd_insert[i];
            delete rnd_get[i];
            delete rnd_scan[i];
            delete rnd_delete[i];
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 6) {
        std::cout << "Usage: ./ycsb [index type] [ycsb workload type] [key distribution] [access pattern] [number of threads]\n";
        std::cout << "1. index type: art hot bwtree masstree clht\n";
        std::cout << "               fastfair levelhash cceh woart\n";
        std::cout << "2. ycsb workload type: a, b, c, e\n";
        std::cout << "3. key distribution: randint, string\n";
        std::cout << "4. access pattern: uniform, zipfian\n";
        std::cout << "5. number of threads (integer)\n";
        return 1;
    }

    printf("%s, workload%s, %s, %s, threads %s\n", argv[1], argv[2], argv[3], argv[4], argv[5]);

    int index_type;
    if (strcmp(argv[1], "art") == 0)
        index_type = TYPE_ART;
    else if (strcmp(argv[1], "hot") == 0) {
#ifdef HOT
        index_type = TYPE_HOT;
#else
        return 1;
#endif
    } else if (strcmp(argv[1], "bwtree") == 0)
        index_type = TYPE_BWTREE;
    else if (strcmp(argv[1], "masstree") == 0)
        index_type = TYPE_MASSTREE;
    else if (strcmp(argv[1], "clht") == 0)
        index_type = TYPE_CLHT;
    else if (strcmp(argv[1], "fastfair") == 0)
        index_type = TYPE_FASTFAIR;
    else if (strcmp(argv[1], "levelhash") == 0)
        index_type = TYPE_LEVELHASH;
    else if (strcmp(argv[1], "cceh") == 0)
        index_type = TYPE_CCEH;
    else if (strcmp(argv[1], "woart") == 0)
        index_type = TYPE_WOART;
    else {
        fprintf(stderr, "Unknown index type: %s\n", argv[1]);
        exit(1);
    }

    int wl;
    if (strcmp(argv[2], "a") == 0) {
        wl = WORKLOAD_A;
    } else if (strcmp(argv[2], "b") == 0) {
        wl = WORKLOAD_B;
    } else if (strcmp(argv[2], "c") == 0) {
        wl = WORKLOAD_C;
    } else if (strcmp(argv[2], "d") == 0) {
        wl = WORKLOAD_D;
    } else if (strcmp(argv[2], "e") == 0) {
        wl = WORKLOAD_E;
    } else {
        fprintf(stderr, "Unknown workload: %s\n", argv[2]);
        exit(1);
    }

    int kt;
    if (strcmp(argv[3], "randint") == 0) {
        kt = RANDINT_KEY;
    } else if (strcmp(argv[3], "string") == 0) {
        kt = STRING_KEY;
    } else {
        fprintf(stderr, "Unknown key type: %s\n", argv[3]);
        exit(1);
    }

    int ap;
    if (strcmp(argv[4], "uniform") == 0) {
        ap = UNIFORM;
    } else if (strcmp(argv[4], "zipfian") == 0) {
        ap = ZIPFIAN;
        fprintf(stderr, "Not supported access pattern: %s\n", argv[4]);
        exit(1);
    } else {
        fprintf(stderr, "Unknown access pattern: %s\n", argv[4]);
        exit(1);
    }

    int num_thread = atoi(argv[5]);
    tbb::task_scheduler_init init(num_thread);

    if (kt != STRING_KEY) {
        std::vector<uint64_t> init_keys;
        std::vector<uint64_t> keys;
        std::vector<int> ranges;
        std::vector<int> ops;

        init_keys.reserve(LOAD_SIZE);
        keys.reserve(RUN_SIZE);
        ranges.reserve(RUN_SIZE);
        ops.reserve(RUN_SIZE);

        memset(&init_keys[0], 0x00, LOAD_SIZE * sizeof(uint64_t));
        memset(&keys[0], 0x00, RUN_SIZE * sizeof(uint64_t));
        memset(&ranges[0], 0x00, RUN_SIZE * sizeof(int));
        memset(&ops[0], 0x00, RUN_SIZE * sizeof(int));

        motivation_run_randint(index_type, wl, kt, ap, num_thread, init_keys, keys, ranges, ops);
    } else {
        printf("No impl for string in motivation\n");
    }

    return 0;
}
