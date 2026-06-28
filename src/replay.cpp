#include "../include/replay.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>

using namespace std;

static vector<string> split(const string& s, char delim) {
    vector<string> parts;
    string item;
    stringstream ss(s);

    while (getline(ss, item, delim)) {
        parts.push_back(item);
    }
    return parts;
}

bool ReplayEngine::saveSession(const string& path, const vector<Metrics>& events) {
    ofstream out(path);
    if (!out.is_open()) return false;

    for (const auto& e : events) {
        out << e.event_id << '|'
            << e.token_id << '|'
            << e.layer_name << '|'
            << e.submodule_name << '|'
            << e.tensor_shape << '|'
            << e.dtype << '|'
            << e.timestamp_ms << '|'
            << e.latency_ms << '|'
            << e.sparsity_rate << '|'
            << e.mean_activation << '|'
            << e.max_activation << '|'
            << (e.anomaly_flag ? 1 : 0) << '\n';
    }

    return true;
}

vector<Metrics> ReplayEngine::loadSession(const string& path) {
    vector<Metrics> events;
    ifstream in(path);
    if (!in.is_open()) return events;

    string line;
    while (getline(in, line)) {
        if (line.empty()) continue;

        auto parts = split(line, '|');
        if (parts.size() != 12) continue;

        Metrics m;
        m.event_id = stoi(parts[0]);
        m.token_id = stoi(parts[1]);
        m.layer_name = parts[2];
        m.submodule_name = parts[3];
        m.tensor_shape = parts[4];
        m.dtype = parts[5];
        m.timestamp_ms = stod(parts[6]);
        m.latency_ms = stod(parts[7]);
        m.sparsity_rate = stod(parts[8]);
        m.mean_activation = stod(parts[9]);
        m.max_activation = stod(parts[10]);
        m.anomaly_flag = (stoi(parts[11]) != 0);

        events.push_back(m);
    }

    return events;
}

void ReplayEngine::replayToBuffer(const vector<Metrics>& events, RingBuffer& rb, int delay_ms) {
    for (const auto& e : events) {
        rb.push(e);
        if (delay_ms > 0) {
            this_thread::sleep_for(chrono::milliseconds(delay_ms));
        }
    }
}