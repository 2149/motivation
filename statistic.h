#pragma once

#include <chrono>
#include <ratio>
#include <iostream>
using namespace std;

// ATTENTION: only for single thread!!!
const static char *prefixs[] = {"PutTest", "GetTest", "DeleteTest", "ScanTest"};

enum {
    PutPrefix = 0,
    GetPrefix,
    DelPrefix,
    ScanPrefix,

};
class Statistic{
public:
    Statistic() {
        read_ = 0.0;
        write_ = 0.0;
        num_ = 0;
        total_num_ = 0;
        total_read_ = 0.0;
        total_write_ = 0.0;
        comp_lat_ = 0.0;
        total_comp_lat_ = 0.0;
        comp_num_ = 0;
        total_comp_num_ = 0;
        split_num_ = 0;
        node_search_ = 0;
        put_ = 0;
        get_ = 0;
        delete_ = 0;
        scan_ = 0;
        function_count = 0;
        get_search_ = 0;
        start_ = end_ = chrono::high_resolution_clock::now();
    }
    ~Statistic() = default;

    void start() {
        start_ = chrono::high_resolution_clock::now();
    }

    void end() {
        end_ = chrono::high_resolution_clock::now();
    }

    void Initail() {
        statisticcstart_ = chrono::high_resolution_clock::now();
    }
    void add_search() {
        chrono::duration<double, std::nano> diff = end_ - start_;
        read_ += diff.count();
        total_read_ += diff.count();
    }

    void add_write() {
        chrono::duration<double, std::nano> diff = end_ - start_;
        write_ += diff.count();
        total_write_ += diff.count();
    }

    void add_comp_lat() {
        chrono::duration<double, std::nano> diff = end_ - start_;
        comp_lat_ += diff.count();
        total_comp_lat_ += diff.count();
    }

    void add_comp_num(){
        comp_num_++;
        total_comp_num_++;
    };

    void add_entries_num() {
        num_++;
        total_num_++;
    }
    
    void add_split_num() {
        split_num_ ++;
    }

    void clear_period() {
        read_ = 0.0;
        write_ = 0.0;
        num_ = 0;
        node_search_ = 0;
        tree_level_ = 0;
        comp_lat_ = 0.0;
        comp_num_ = 0;
        split_num_ = 0;
        put_ = 0;
        get_ = 0;
        delete_ = 0;
        scan_ = 0;
        function_count = 0;
    }

    void add_node_search(){node_search_++;}

    void add_tree_level(int treeLevel) {tree_level_ += treeLevel;}

    void print_cur(){
        chrono::duration<double, std::nano> diff = end_ - start_;
        chrono::duration<double, std::nano> diff_start = start_ - statisticcstart_;
        chrono::duration<double, std::nano> diff_end = end_ - statisticcstart_;
        printf("total_time: %lf s, start %lf s, end %lf s\n", 
                diff.count() * 1e-9, diff_start.count() * 1e-9, diff_end.count() * 1e-9);
        // cout<<"total_time: "<<diff.count() * 1e-9<<"s\n";
    }

    void print_spilt() {
        // printf("tatol node %lld, spilt node %lld, with %lf.\n", 
        printf("tatol node %lld, spilt node %lld.\n", 
                total_num_, split_num_);
    }

    void print_put() {
        if(num_  > 0)
        cout
        <<"num "<<num_
        <<" period_put_search_latency(s) "<< read_ * 1e-9
        <<" average_put_search_latency(ns) "<< read_ / num_
        <<" period_put_write_latency(ns) "<< write_ * 1e-9
        <<" average_put_write_latency(ns) "<< write_ / num_
        <<"\n";
    }

    void print_get() {
        if(function_count  > 0)
        cout
        <<"num "<<function_count
        <<" period_get_search_latency(s) "<< get_search_ * 1e-9
        <<" average_get_search_latency(ns) "<< get_search_ / function_count
        <<" period_get_write_latency(ns) "<< get_ * 1e-9
        <<" average_get_write_latency(ns) "<< get_ / function_count
        <<"\n";
    }

    void add_put() {
        function_count ++;
        chrono::duration<double, std::nano> diff = end_ - start_;
        put_ += diff.count();
    }

    void add_get_search() {
        function_count ++;
        chrono::duration<double, std::nano> diff = end_ - start_;
        get_search_ += diff.count();
    }

    void add_get() {
        function_count ++;
        chrono::duration<double, std::nano> diff = end_ - start_;
        get_ += diff.count();
    }

    void add_delete() {
        function_count ++;
        chrono::duration<double, std::nano> diff = end_ - start_;
        delete_ += diff.count();
    }

    void add_scan() {
        function_count ++;
        chrono::duration<double, std::nano> diff = end_ - start_;
        scan_ += diff.count();
    }

    void print_latency() {
        if(function_count == 0) {
            return ;
        }
        cout
        // //<<"num "<<num_
        // //<<" period_read_latency(ns) "<<read_
        // <<"average_read_latency(ns) "<<read_ / num_
        // <<" average_node_search "<<node_search_ / num_
        // <<" average_tree_level "<<tree_level_ / num_
        // //<<" period_write_latency(ns) "<<write_
        // <<" average_write_latency(ns) "<<write_ / num_
        // // <<" split_times "<<split_num_  
        // <<" split_times "<<split_num_ 
        // //<<" average_compare_latency(ns) "<<comp_lat_ / comp_num_
        // //<<" average_compare_times "<<comp_num_ / num_
        //<<"num "<<num_
        //<<" period_read_latency(ns) "<<read_
        <<" average_put_latency(us) "<<put_ * 1e-3 / function_count
        <<" average_get_latency(us) "<<get_ * 1e-3 / function_count
        <<" average_delete_latency(us) "<<delete_ * 1e-3 / function_count
        <<" average_scan_latency(us) "<<scan_ * 1e-3/ function_count
        <<"\n";
    }
    void PrintLatency(uint64_t i, int Prefix = PutPrefix) {
        cout << prefixs[Prefix] << ":" << i;
        print_latency();
        clear_period();
    }
private:
    double read_;
    double write_;
    double total_read_;
    double total_write_;
    double get_search_;
    double get_read_;

    double comp_lat_;
    double total_comp_lat_;
    uint64_t comp_num_;
    uint64_t total_comp_num_;

    double put_;
    double get_;
    double delete_;
    double scan_;
    uint64_t function_count;

    chrono::high_resolution_clock::time_point start_;
    chrono::high_resolution_clock::time_point end_;
    chrono::high_resolution_clock::time_point statisticcstart_;
    uint64_t num_;
    uint64_t total_num_;
    uint64_t split_num_;

    uint64_t node_search_;
    uint64_t tree_level_;
};