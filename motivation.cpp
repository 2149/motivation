
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
static uint64_t RUN_SIZE = 100000;

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

void motivation_run_randint(int index_type, int num_thread)
{
    Statistic stats;


    rocksdb::Random64 *rnd_insert[num_thread];
    rocksdb::Random64 *rnd_get[num_thread];
    rocksdb::Random64 *rnd_scan[num_thread];
    rocksdb::Random64 *rnd_delete[num_thread];

    std::atomic<int> range_complete, range_incomplete;
    std::atomic<int> notfound;
    range_complete.store(0);
    range_incomplete.store(0);
    notfound.store(0);

    {
        for(int i = 0; i <= num_thread; i ++) {
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
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = LOAD_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + LOAD_SIZE / num_thread;
                auto t = tree.getThreadInfo();

                // printf("Thread id = %d, start %lld, end %lld.\n", thread_id, start_key, end_key);
                for (uint64_t i = start_key; i != end_key; i++) {
                    uint64_t key_64 = rnd_insert[thread_id]->Next();
                    Key *key = key->make_leaf(key_64, sizeof(uint64_t), key_64);
                    stats.start();
                    tree.insert(key, t);
                    stats.end();
                    stats.add_put();
                    if ((i % 10000) == 0) {
                        stats.PrintLatency(i);
                        ART_ROWEX::print_stats(10000);
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

        {
            // Put
            Key *end = end->make_leaf(UINT64_MAX, sizeof(uint64_t), 0);
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = RUN_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + RUN_SIZE / num_thread;
                auto t = tree.getThreadInfo();

                for (uint64_t i = start_key; i != end_key; i++) {
                    uint64_t key_64 = rnd_insert[thread_id]->Next();
                    Key *key = key->make_leaf(key_64, sizeof(uint64_t), key_64);
                    tree.insert(key, t);
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

        {
            // Get
            Key *end = end->make_leaf(UINT64_MAX, sizeof(uint64_t), 0);
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = RUN_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + RUN_SIZE / num_thread;
                auto t = tree.getThreadInfo();
                
                for (uint64_t i = start_key; i != end_key; i++) {
                    uint64_t key_64 = rnd_get[thread_id]->Next();
                    Key *key = key->make_leaf(key_64, sizeof(uint64_t), 0);
                    uint64_t *val = reinterpret_cast<uint64_t *>(tree.lookup(key, t));
                    if (*val != key_64) {
                        std::cout << "[ART] wrong key read: " << val << " expected:" << key_64 << std::endl;
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

        {
            // Scan
            Key *end = end->make_leaf(UINT64_MAX, sizeof(uint64_t), 0);
            int count = 4;
            int scan_times = 100;
            int scan_count = 100;
            while(count >0 ) {
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = scan_times / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + scan_times / num_thread;
                auto t = tree.getThreadInfo();

                for (uint64_t i = start_key; i != end_key; i++) {
                    uint64_t key_64 = rnd_scan[thread_id]->Next();
                    Key *results[scan_count];
                    Key *continueKey = NULL;
                    size_t resultsFound = 0;
                    size_t resultsSize = scan_count;
                    Key *start = start->make_leaf(key_64, sizeof(uint64_t), 0);
                    tree.lookupRange(start, end, continueKey, results, resultsSize, resultsFound, t);
                }
            };
            std::vector<std::thread> thread_group;

            for (int i = 0; i < num_thread; i++)
                thread_group.push_back(std::thread{func});

            for (int i = 0; i < num_thread; i++)
                thread_group[i].join();
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
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = RUN_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + RUN_SIZE / num_thread;
                auto t = tree.getThreadInfo();
                for (uint64_t i = start_key; i != end_key; i++) {
                    uint64_t key_64 = rnd_delete[thread_id]->Next();
                    Key *key = key->make_leaf(key_64, sizeof(uint64_t), 0);
                    tree.remove(key, t);
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
#ifdef HOT
    } else if (index_type == TYPE_HOT) {
        // printf("No implie for HOT with motivation!\n");
        printf("Motivation test HOT start!\n");
        hot::rowex::HOTRowex<IntKeyVal *, IntKeyExtractor> mTrie;
        {
            // Load
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = LOAD_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + LOAD_SIZE / num_thread;
                for (uint64_t i = start_key; i != end_key; i++) {
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
                    if ((i % 10000) == 0) {
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

        {
            // Put
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = RUN_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + RUN_SIZE / num_thread;
                for (uint64_t i = start_key; i != end_key; i++) {
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

        {
            // Get
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = RUN_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + RUN_SIZE / num_thread;
                for (uint64_t i = start_key; i != end_key; i++) {
                    uint64_t key_64 = rnd_get[thread_id]->Next();
                    idx::contenthelpers::OptionalValue<IntKeyVal *> result = mTrie.lookup(key_64);
                    if (!result.mIsValid || result.mValue->value != key_64) {
                        notfound ++;
                        // printf("mIsValid = %d\n", result.mIsValid);
                        // printf("Return value = %lu, Correct value = %lu\n", result.mValue->value, key_64);
                        // exit(1);
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
            printf("Not found key %d\n", notfound.load());
        }

        {
            // Scan
            int count = 4;
            int scan_times = 100;
            int scan_count = 100;
            
            while(count > 0) {
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = scan_times / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + scan_times / num_thread;
                for (uint64_t i = start_key; i != end_key; i++) {
                    uint64_t key_64 = rnd_scan[thread_id]->Next();
                    uintptr_t buf[scan_count];
                    hot::rowex::HOTRowexSynchronizedIterator<IntKeyVal *, IntKeyExtractor> it = mTrie.lower_bound(key_64);
                    int resultsFound = 0;
                    while (it != mTrie.end() && resultsFound < scan_count) {
                        buf[resultsFound] = (*it)->value;
                        resultsFound++;
                        ++it;
                    }
                    // printf("Found %d, while scan %d\n", resultsFound, scan_count);
                    // idx::contenthelpers::OptionalValue<IntKeyVal *> result = mTrie.scan(key_64, scan_count);
                }

            };
            std::vector<std::thread> thread_group;

            for (int i = 0; i < num_thread; i++)
                thread_group.push_back(std::thread{func});

            for (int i = 0; i < num_thread; i++)
                thread_group[i].join();

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
                    if ((i % 10000) == 0) {
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
                    uint64_t key_64 = rnd_delete[thread_id]->Next();
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
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = LOAD_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + LOAD_SIZE / num_thread;
                for (uint64_t i = start_key; i != end_key; i++) {
                    uint64_t key_64 = rnd_insert[thread_id]->Next();
                    stats.start();
                    tree->put(key_64, (void *)key_64);
                    stats.end();
                    stats.add_put();
                    if ((i % 10000) == 0) {
                        stats.PrintLatency(i);
                        masstree::print_stats(10000);
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

        {
            // Put
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = RUN_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + RUN_SIZE / num_thread;
                for (uint64_t i = start_key; i != end_key; i++) {
                    uint64_t key_64 = rnd_insert[thread_id]->Next();
                    tree->put(key_64, (void *)key_64);
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

        {
            // Get
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = RUN_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + RUN_SIZE / num_thread;
                for (uint64_t i = start_key; i != end_key; i++) { 
                    uint64_t key_64 = rnd_get[thread_id]->Next();
                    uint64_t *ret = reinterpret_cast<uint64_t *> (tree->get(key_64));
                    if ((uint64_t)ret != key_64) {
                        printf("[MASS] search key = %lu, search value = %lu\n", key_64, (uint64_t)ret);
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

        {
            // Scan
            Key *end = end->make_leaf(UINT64_MAX, sizeof(uint64_t), 0);
            int count = 4;
            int scan_times = 100;
            int scan_count = 100;
            while(count >0 ) {
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = scan_times / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + scan_times / num_thread;
                for (uint64_t i = start_key; i != end_key; i++) {
                    uint64_t key_64 = rnd_scan[thread_id]->Next();
                    uint64_t buf[scan_count];
                    int ret = tree->scan(key_64, scan_count, buf);
                }
            };
            std::vector<std::thread> thread_group;

            for (int i = 0; i < num_thread; i++)
                thread_group.push_back(std::thread{func});

            for (int i = 0; i < num_thread; i++)
                thread_group[i].join();
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
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = RUN_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + RUN_SIZE / num_thread;
                for (uint64_t i = start_key; i != end_key; i++) {
                    uint64_t key_64 = rnd_delete[thread_id]->Next();
                    tree->del(key_64);
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
                    if ((i % 10000) == 0) {
                        stats.PrintLatency(i);
                        clht_print_stats(10000);
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
        // printf("No implie for MASSTREE with motivation!\n");
        printf("Motivation test Fast-Fair start!\n");
        fastfair::btree *bt = new fastfair::btree();
        {
            // Load
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = LOAD_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + LOAD_SIZE / num_thread;

                // printf("Thread id = %d, start %lld, end %lld.\n", thread_id, start_key, end_key);
                for (uint64_t i = start_key; i != end_key; i++) {
                    uint64_t key_64 = rnd_insert[thread_id]->Next();
                    stats.start();
                    bt->btree_insert(key_64, (char *)key_64);
                    stats.end();
                    stats.add_put();
                    if ((i % 10000) == 0) {
                        stats.PrintLatency(i);
                        fastfair::print_stats(10000);
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

        {
            // Put
            Key *end = end->make_leaf(UINT64_MAX, sizeof(uint64_t), 0);
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = RUN_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + RUN_SIZE / num_thread;
                for (uint64_t i = start_key; i != end_key; i++) {
                    uint64_t key_64 = rnd_insert[thread_id]->Next();
                    bt->btree_insert(key_64, (char *)key_64);
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

        {
            // Get
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = RUN_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + RUN_SIZE / num_thread;
                
                for (uint64_t i = start_key; i != end_key; i++) {
                    uint64_t key_64 = rnd_get[thread_id]->Next();
                    uint64_t *val = reinterpret_cast<uint64_t *>(bt->btree_search(key_64));
                    if ((uint64_t)val != key_64) {
                        notfound ++;
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
            printf("Not found key %d\n", notfound.load());
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
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = scan_times / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + scan_times / num_thread;

                for (uint64_t i = start_key; i != end_key; i++) {
                    uint64_t key_64 = rnd_scan[thread_id]->Next();
                    int resultsFound = 0;
                    int resultsSize = scan_count;
                    uint64_t results[resultsSize];
                    bt->btree_search_range (key_64, UINT64_MAX, results, resultsSize, resultsFound);
                }
            };
            std::vector<std::thread> thread_group;

            for (int i = 0; i < num_thread; i++)
                thread_group.push_back(std::thread{func});

            for (int i = 0; i < num_thread; i++)
                thread_group[i].join();
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
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = RUN_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + RUN_SIZE / num_thread;
                for (uint64_t i = start_key; i != end_key; i++) {
                    uint64_t key_64 = rnd_delete[thread_id]->Next();
                    bt->btree_delete(key_64);
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
        delete bt;
    } else if (index_type == TYPE_LEVELHASH) {
        // printf("No implie for MASSTREE with motivation!\n");
        printf("Motivation test Level-Hash start!\n");
        Hash *table = new LevelHashing(10);
        {
            // Load
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = LOAD_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + LOAD_SIZE / num_thread;
                for (uint64_t i = start_key; i != end_key; i++) {
                    uint64_t key_64 = rnd_insert[thread_id]->Next();
                    stats.start();
                    table->Insert(key_64, reinterpret_cast<const char*>(key_64));
                    stats.end();
                    stats.add_put();
                    if ((i % 10000) == 0) {
                        stats.PrintLatency(i);
                        lh_print_stats(10000);
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

        {
            // Put
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = RUN_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + RUN_SIZE / num_thread;
                for (uint64_t i = start_key; i != end_key; i++) {
                    uint64_t key_64 = rnd_insert[thread_id]->Next();
                    table->Insert(key_64, reinterpret_cast<const char*>(key_64));
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

        {
            // Get
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = RUN_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + RUN_SIZE / num_thread;
                for (uint64_t i = start_key; i != end_key; i++) { 
                    uint64_t key_64 = rnd_get[thread_id]->Next();
                    auto val = table->Get(key_64);
                    if (val == NONE || ((uint64_t)val) != key_64) {
                        std::cout << "[Level Hashing] wrong key read: " << *(uint64_t *)val << " expected: " << key_64 << std::endl;
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

        {
            // Delete
            next_thread_id.store(0);
            auto starttime = std::chrono::system_clock::now();
            auto func = [&]() {
                int thread_id = next_thread_id.fetch_add(1);
                uint64_t start_key = RUN_SIZE / num_thread * (uint64_t)thread_id;
                uint64_t end_key = start_key + RUN_SIZE / num_thread;
                for (uint64_t i = start_key; i != end_key; i++) {
                    uint64_t key_64 = rnd_delete[thread_id]->Next();
                    table->Delete(key_64);
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
        delete table;

    } else if (index_type == TYPE_CCEH) {
        printf("No implie for MASSTREE with motivation!\n");
    } else if (index_type == TYPE_WOART) {
        printf("No implie for MASSTREE with motivation!\n");
#ifndef STRING_TYPE
        
#endif
    }

    {
        for(int i = 0; i <= num_thread; i ++) {
            delete rnd_insert[i];
            delete rnd_get[i];
            delete rnd_scan[i];
            delete rnd_delete[i];
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cout << "Usage: ./ycsb [index type] [number of threads] (load size) (run size)\n";
        std::cout << "1. index type: art hot bwtree masstree clht\n";
        std::cout << "               fastfair levelhash cceh woart\n";
        std::cout << "2. number of threads (integer)\n";
        std::cout << "[3]. load size (integer)\n";
        std::cout << "[4]. run size (integer)\n";
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

    int num_thread = atoi(argv[2]);

    if(argc > 3) {
        LOAD_SIZE = atoi(argv[3]);
    }

    if(argc > 4) {
        RUN_SIZE = atoi(argv[4]);
    }
    printf("Load size: %d, Run size %d\n", LOAD_SIZE, RUN_SIZE);

    tbb::task_scheduler_init init(num_thread);
    
    motivation_run_randint(index_type, num_thread);
    

    return 0;
}
