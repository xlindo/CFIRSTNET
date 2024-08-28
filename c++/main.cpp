#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <map>
#include <cmath>
#include <stdlib.h>

#include <boost/algorithm/string.hpp>
#include <boost/bimap.hpp>
#include <boost/multi_array.hpp>
#include <boost/lexical_cast.hpp>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/multi_point.hpp>
#include <boost/geometry/geometries/multi_polygon.hpp>

typedef boost::geometry::model::d2::point_xy<float> point_type;
typedef boost::geometry::model::linestring<point_type> linestring_type;

typedef boost::geometry::model::multi_linestring<linestring_type> multi_linestring_type;
typedef boost::geometry::model::multi_point<point_type> multi_point_type;

#define VERTICAL   1
#define HORIZONTAL 2

using namespace std;

// disclamer: x is for height coordinate, y is for width coordinate
struct coordinate{
    int x;
    int y;
};

inline bool operator==(const coordinate& c1, const coordinate& c2) {
    return c1.x == c2.x && c1.y == c2.y;
}

struct wire_cmp{
    bool operator()(const string& a, const string& b) const { return stoi(a.substr(1)) > stoi(b.substr(1)); };
};

struct partition_cmp{
    bool operator()(const string& a, const string& b) const { return stoi(a.substr(1)) < stoi(b.substr(1)); };
};

struct node {
    coordinate c;
    int R1; // distance to front via
    int R2; // distance to back via
    float current = 0.0; // x1 <- node
    float resistance = 0.0;
    
    vector<float> *ir_drop_ptr = new vector<float>(9, 0.0);
    float *current_source = new float(0.0);
};

struct wire {
    coordinate c1, c2; // m1 < m2, x1 < x2, y1 < y2
    float resistance;
    float resistance_per_unit;
};

struct h_wire {
    int h, w1, w2;
};

struct v_wire {
    int w, h1, h2;
};

struct local_partition {
    bool v1, v2;
    float resistance_per_unit;

    vector<node> nodes;
};

bool in_range(coordinate c, wire w) {
    return w.c1.x <= c.x && c.x <= w.c2.x && w.c1.y <= c.y && c.y <= w.c2.y;
}

void sort_node_horizontal(vector<node>& nodes) {
    sort(nodes.begin(), nodes.end(), [](node a, node b) {
        if (a.c.x != b.c.x) return a.c.x < b.c.x;
        if (a.c.y != b.c.y) return a.c.y < b.c.y;
        return false;
    });
}

void sort_node_vertical(vector<node>& nodes) {
    sort(nodes.begin(), nodes.end(), [](node a, node b) {
        if (a.c.y != b.c.y) return a.c.y < b.c.y;
        if (a.c.x != b.c.x) return a.c.x < b.c.x;
        return false;
    });
}

void sort_wire_horizontal(vector<wire>& wires) {
    sort(wires.begin(), wires.end(), [](wire a, wire b) {
        if (a.c1.x != b.c1.x) return a.c1.x < b.c1.x;
        if (a.c1.y != b.c1.y) return a.c1.y < b.c1.y;
        return false;
    });
}

void sort_wire_vertical(vector<wire>& wires) {
    sort(wires.begin(), wires.end(), [](wire a, wire b) {
        if (a.c1.y != b.c1.y) return a.c1.y < b.c1.y;
        if (a.c1.x != b.c1.x) return a.c1.x < b.c1.x;
        return false;
    });
}

std::vector<std::string_view> SplitStringV2(std::string_view input, char delimiter) {
    std::vector<std::string_view> tokens;
    const char* token_start = input.data();
    const char* p = token_start;
    const char* end_pos = input.data() + input.size();
    for (; p != end_pos; ++p) {
        if (*p == delimiter) {
        if (p > token_start) {
            tokens.emplace_back(token_start, p - token_start);
        }
        token_start = p + 1;
        continue;
        }
    }
    if (p > token_start) {
        tokens.emplace_back(token_start, p - token_start);
    }
    return tokens;
}

std::vector<std::string_view> MultiSplitStringV2(std::string_view input, std::string_view delimiters) {
    if (delimiters.empty()) {
        return {input};
    }
    std::vector<std::string_view> tokens;
    const char* token_start = input.data();
    const char* p = token_start;
    const char* end_pos = input.data() + input.size();
    for (; p != end_pos; ++p) {
        bool match_delimiter = false;
        for (auto delimiter : delimiters) {
            if (*p == delimiter) {
                match_delimiter = true;
                break;
            }
        }
        if (match_delimiter) {
            if (p > token_start) {
                tokens.emplace_back(token_start, p - token_start);
            }
            token_start = p + 1;
            continue;
        }
    }
    if (p > token_start) {
        tokens.emplace_back(token_start, p - token_start);
    }
    return tokens;
}

class circuit {
    protected:
        boost::bimap<string, int> metal_layer;
        boost::bimap<string, int> via_layer;
        vector<int> orientation;

        vector<vector<node>> V;
        vector<vector<node>> I;
        vector<vector<wire>> R;
        vector<vector<wire>> Via;

        vector<vector<node>> current_map;
        vector<vector<node>> voltage_map;
        vector<vector<wire>> wire_map;
        vector<vector<local_partition> > partition_map;

        int x_max = 0, y_max = 0;
        int H, W;

    public:
        void set_orientation(vector<string> metal, vector<int> orientation) {
            for (int i = 0; i < metal.size(); i++) {
                metal_layer.insert({metal[i], i});
                this->orientation.push_back(orientation[i]);
            }

            for (int i = 0; i < metal.size() - 1; i++) {
                via_layer.insert({metal[i] + metal[i + 1], i});
            }
        }

        void set_via(vector<string> metal, vector<int> orientation) {
            for (int i = 0; i < metal.size(); i++) {
                metal_layer.insert({metal[i], i});
                this->orientation.push_back(orientation[i]);
            }

            for (int i = 0; i < metal.size() - 1; i++) {
                via_layer.insert({metal[i] + metal[i + 1], i});
            }
        }

        void set_size(int h, int w) {
            H = h;
            W = w;
        }

        void read_data(string filename) {
            ifstream infile;
            infile.open(filename);

            R.resize(metal_layer.size());
            I.resize(metal_layer.size());
            V.resize(metal_layer.size());
            Via.resize(via_layer.size());

            string line;
            while (getline(infile, line)) {
                auto str_vec = MultiSplitStringV2(line, " _");

                if (str_vec[0][0] == 'R') {
                    wire w;
                    w.c1.x = atoi(str_vec[3].data());
                    w.c1.y = atoi(str_vec[4].data());
                    w.c2.x = atoi(str_vec[7].data());
                    w.c2.y = atoi(str_vec[8].data());
                    w.resistance = atof(str_vec[9].data());

                    auto m1 = string(str_vec[2]);
                    auto m2 = string(str_vec[6]);

                    if (str_vec[2] != str_vec[6]) {
                        if(stoi(m1.substr(1)) > stoi(m2.substr(1))) {
                            swap(w.c1, w.c2);
                        }
                        Via[via_layer.left.at(m1+m2)].push_back(w);
                    }
                    else if (w.c1.x != w.c2.x) {
                        if (w.c1.x > w.c2.x) {
                            swap(w.c1, w.c2);
                        }
                        R[metal_layer.left.at(m1)].push_back(w); //m1, m7, m9
                    }
                    else if (w.c1.y != w.c2.y) {
                        if (w.c1.y > w.c2.y) {
                            swap(w.c1, w.c2);
                        }
                        R[metal_layer.left.at(m1)].push_back(w); //m4, m8
                    }

                    if (w.c2.x > x_max) 
                        x_max = w.c2.x;
                    
                    if (w.c2.y > y_max) 
                        y_max = w.c2.y;
                }
                else if (str_vec[0][0] == 'I') {
                    node n;
                    n.c.x = atoi(str_vec[3].data());
                    n.c.y = atoi(str_vec[4].data());
                    *n.current_source = atof(str_vec[6].data());

                    auto m1 = string(str_vec[2]);

                    I[metal_layer.left.at(m1)].push_back(n);
                }
                else if (str_vec[0][0] == 'V') {
                    node n;
                    n.c.x = atoi(str_vec[3].data());
                    n.c.y = atoi(str_vec[4].data());

                    auto m1 = string(str_vec[2]);

                    V[metal_layer.left.at(m1)].push_back(n);
                }
            }
        }

        void merge_wires() {
            wire_map.resize(R.size());

            for(int metal = 0; metal < metal_layer.size(); metal++) {
                vector<wire> &r = R[metal];

                if (orientation[metal] == VERTICAL) {
                    sort_wire_vertical(r);
                }
                else if (orientation[metal] == HORIZONTAL) {
                    sort_wire_horizontal(r);
                }

                vector<wire> new_wires;
                wire w = r[0];
                for (int i = 1; i < r.size(); i++) {
                    if(w.c2 == r[i].c1){
                        w.c2 = r[i].c2;
                        w.resistance += r[i].resistance;
                    }
                    else{
                        w.resistance_per_unit = w.resistance / (w.c2.x - w.c1.x + w.c2.y - w.c1.y);
                        new_wires.push_back(w);
                        w = r[i];
                    }
                }
                w.resistance_per_unit = w.resistance / (w.c2.x - w.c1.x + w.c2.y - w.c1.y);
                new_wires.push_back(w);

                wire_map[metal] = new_wires;
            }
        }
        
        void merge_nodes() {
            current_map.resize(metal_layer.size());
            for (int metal = 0; metal < metal_layer.size(); metal++) {
                for (auto& n : I[metal]) {
                    current_map[metal].push_back(n);
                }
            }

            for(int metal = 0; metal < via_layer.size(); metal++) {
                for (auto& w : Via[metal]) {
                    node n;
                    if(metal % 2 == 0)
                        n.c = w.c2;
                    else
                        n.c = w.c1;
                    n.resistance = w.resistance;
                    current_map[metal + 1].push_back(n);
                }
            }
            voltage_map = V;

            for (int metal = 0; metal < metal_layer.size(); metal++) {
                if (orientation[metal] == VERTICAL) {
                    sort_node_vertical(current_map[metal]);
                }
                else if (orientation[metal] == HORIZONTAL) {
                    sort_node_horizontal(current_map[metal]);
                }
            }
        }

        void get_partition() {
            partition_map.resize(metal_layer.size());

            for(int metal = metal_layer.size() - 1; metal >= 0; metal--) {
                vector<wire> &wires = wire_map[metal];
                vector<node> &current_nodes = current_map[metal];

                bool is_voltage;
                int current_index = 0;
                int voltage_index = 0;

                if (orientation[metal] == VERTICAL) {
                    sort_node_vertical(voltage_map[metal]);
                    vector<node> &voltage_nodes = voltage_map[metal];

                    for(auto& w : wires) {
                        while(true) {
                            if(voltage_nodes.size() == voltage_index && current_nodes.size() == current_index) {
                                break;
                            }
                            node n;

                            if(voltage_nodes.size() == voltage_index) {
                                n = current_nodes[current_index];
                                is_voltage = false;
                            }
                            else if(current_nodes.size() == current_index) {
                                n = voltage_nodes[voltage_index];
                                is_voltage = true;
                            }
                            else if(voltage_nodes[voltage_index].c.y < current_nodes[current_index].c.y) {
                                n = voltage_nodes[voltage_index];
                                is_voltage = true;
                            }
                            else if(voltage_nodes[voltage_index].c.y > current_nodes[current_index].c.y) {
                                n = current_nodes[current_index];
                                is_voltage = false;
                            }
                            else {
                                if(voltage_nodes[voltage_index].c.x < current_nodes[current_index].c.x) {
                                    n = voltage_nodes[voltage_index];
                                    is_voltage = true;
                                }
                                else {
                                    n = current_nodes[current_index];
                                    is_voltage = false;
                                }
                            }

                            if(!in_range(n.c, w)) {
                                if (n.c.y < w.c1.y || (n.c.y == w.c1.y && n.c.x < w.c1.x)) {
                                    if (is_voltage) {
                                        voltage_index++;
                                    }
                                    else {
                                        current_index++;
                                    }
                                    continue;
                                }
                                else {
                                    break;
                                }
                            }

                            local_partition p;
                            p.resistance_per_unit = w.resistance_per_unit;
                            p.v1 = is_voltage;

                            while(true) {
                                p.nodes.push_back(n);
                                p.v2 = is_voltage;

                                if (is_voltage)
                                    voltage_index++;
                                else
                                    current_index++;

                                if(voltage_nodes.size() == voltage_index && current_nodes.size() == current_index) {
                                    break;
                                }
                                else if(voltage_nodes.size() == voltage_index) {
                                    n = current_nodes[current_index];
                                    is_voltage = false;
                                }
                                else if(current_nodes.size() == current_index) {
                                    n = voltage_nodes[voltage_index];
                                    is_voltage = true;
                                }
                                else if(voltage_nodes[voltage_index].c.y < current_nodes[current_index].c.y) {
                                    n = voltage_nodes[voltage_index];
                                    is_voltage = true;
                                }
                                else if(voltage_nodes[voltage_index].c.y > current_nodes[current_index].c.y) {
                                    n = current_nodes[current_index];
                                    is_voltage = false;
                                }
                                else {
                                    if(voltage_nodes[voltage_index].c.x < current_nodes[current_index].c.x) {
                                        n = voltage_nodes[voltage_index];
                                        is_voltage = true;
                                    }
                                    else {
                                        n = current_nodes[current_index];
                                        is_voltage = false;
                                    }
                                }


                                if(in_range(n.c, w)) {
                                    if(is_voltage) {
                                        p.nodes.push_back(n);
                                        p.v2 = is_voltage;

                                        break;
                                    }
                                }
                                else {
                                    break;
                                }
                            }
                            if(p.nodes.size() > 1) {
                                if(p.v1 || p.v2) {
                                    partition_map[metal].push_back(p);
                                    if (metal != 0)
                                        voltage_map[metal - 1].insert(voltage_map[metal - 1].end(), p.nodes.begin() + p.v1, p.nodes.end() - p.v2);
                                }
                            }
                        }
                    }

                }
                else if (orientation[metal] == HORIZONTAL) {
                    sort_node_horizontal(voltage_map[metal]);
                    vector<node> &voltage_nodes = voltage_map[metal];

                    for(auto& w : wires) {
                        while(true) {
                            if(voltage_nodes.size() == voltage_index && current_nodes.size() == current_index) {
                                break;
                            }
                            node n;

                            if(voltage_nodes.size() == voltage_index) {
                                n = current_nodes[current_index];
                                is_voltage = false;
                            }
                            else if(current_nodes.size() == current_index) {
                                n = voltage_nodes[voltage_index];
                                is_voltage = true;
                            }
                            else if(voltage_nodes[voltage_index].c.x < current_nodes[current_index].c.x) {
                                n = voltage_nodes[voltage_index];
                                is_voltage = true;
                            }
                            else if(voltage_nodes[voltage_index].c.x > current_nodes[current_index].c.x) {
                                n = current_nodes[current_index];
                                is_voltage = false;
                            }
                            else {
                                if(voltage_nodes[voltage_index].c.y < current_nodes[current_index].c.y) {
                                    n = voltage_nodes[voltage_index];
                                    is_voltage = true;
                                }
                                else {
                                    n = current_nodes[current_index];
                                    is_voltage = false;
                                }
                            }

                            if(!in_range(n.c, w)) {
                                if (n.c.x < w.c1.x || (n.c.x == w.c1.x && n.c.y < w.c1.y)) {
                                    if (is_voltage) {
                                        voltage_index++;
                                    }
                                    else {
                                        current_index++;
                                    }
                                    continue;
                                }
                                else {
                                    break;
                                }
                            }

                            local_partition p;
                            p.resistance_per_unit = w.resistance_per_unit;
                            p.v1 = is_voltage;

                            while(true) {
                                p.nodes.push_back(n);
                                p.v2 = is_voltage;

                                if (is_voltage)
                                    voltage_index++;
                                else
                                    current_index++;
                                
                                if(voltage_nodes.size() == voltage_index && current_nodes.size() == current_index) {
                                    break;
                                }
                                else if(voltage_nodes.size() == voltage_index) {
                                    n = current_nodes[current_index];
                                    is_voltage = false;
                                }
                                else if(current_nodes.size() == current_index) {
                                    n = voltage_nodes[voltage_index];
                                    is_voltage = true;
                                }
                                else if(voltage_nodes[voltage_index].c.x < current_nodes[current_index].c.x) {
                                    n = voltage_nodes[voltage_index];
                                    is_voltage = true;
                                }
                                else if(voltage_nodes[voltage_index].c.x > current_nodes[current_index].c.x) {
                                    n = current_nodes[current_index];
                                    is_voltage = false;
                                }
                                else {
                                    if(voltage_nodes[voltage_index].c.y < current_nodes[current_index].c.y) {
                                        n = voltage_nodes[voltage_index];
                                        is_voltage = true;
                                    }
                                    else {
                                        n = current_nodes[current_index];
                                        is_voltage = false;
                                    }
                                }

                                if(in_range(n.c, w)) {
                                    if(is_voltage) {
                                        p.nodes.push_back(n);
                                        p.v2 = is_voltage;
                                        
                                        break;
                                    }
                                }
                                else {
                                    break;
                                }
                            }

                            if(p.nodes.size() > 1) {
                                if(p.v1 || p.v2) {
                                    partition_map[metal].push_back(p);
                                    if (metal != 0)
                                        voltage_map[metal - 1].insert(voltage_map[metal - 1].end(), p.nodes.begin() + p.v1, p.nodes.end() - p.v2);
                                }
                            }
                        }
                    }
                }
            }
        }

        void update_wires() {
            wire_map.clear();
            wire_map.resize(partition_map.size());

            for(int metal = 0; metal < metal_layer.size(); metal++) {
                vector<local_partition> &part = partition_map[metal];

                vector<wire> new_wires;
                wire w;
                w.c1 = part.front().nodes.front().c;
                w.c2 = part.front().nodes.back().c;
                w.resistance_per_unit = part.front().resistance_per_unit;
                for (int i = 1; i < part.size(); i++) {
                    if(w.c2 == part[i].nodes.front().c){
                        w.c2 = part[i].nodes.back().c;
                    }
                    else{
                        new_wires.push_back(w);
                        w.c1 = part[i].nodes.front().c;
                        w.c2 = part[i].nodes.back().c;
                        w.resistance_per_unit = part[i].resistance_per_unit;
                    }
                }
                new_wires.push_back(w);

                wire_map[metal] = new_wires;
            }
        }
        
        void get_currents() {
            for (int metal = 0; metal < metal_layer.size(); metal++) {
                vector<local_partition>& partitions = partition_map[metal];

                if (orientation[metal] == VERTICAL) {
                    for (auto& p : partitions) {
                        int x1 = p.nodes.front().c.x;
                        int x2 = p.nodes.back().c.x;

                        for (int i = p.v1; i < p.nodes.size() - p.v2; i++) {
                            int R1 = p.nodes[i].c.x - x1; // x1 <- node
                            int R2 = x2 - p.nodes[i].c.x; // node -> x2

                            p.nodes[i].R1 = R1;
                            p.nodes[i].R2 = R2;

                            int R = R1 + R2;

                            float I1;
                            float I2;

                            if (p.v1 && !p.v2) {
                                I1 = *(p.nodes[i].current_source);
                                I2 = 0.0;
                            }
                            else if (!p.v1 && p.v2) {
                                I1 = 0.0;
                                I2 = *(p.nodes[i].current_source);
                            }
                            else {
                                I1 = *(p.nodes[i].current_source) * R2 / R; // y1 <- node
                                I2 = *(p.nodes[i].current_source) * R1 / R; // node -> y2
                            }

                            for (int j = 0; j <= i; j++) {
                                p.nodes[j].current += I1;
                            }
                            for (int j = i + 1; j < p.nodes.size(); j++) {
                                p.nodes[j].current -= I2;
                            }

                        }
                        if (p.v1) {
                            *(p.nodes.front().current_source) += abs(p.nodes.front().current);
                        }

                        if (p.v2) {
                            *(p.nodes.back().current_source) += abs(p.nodes.back().current);
                        }
                    }
                }
                else if (orientation[metal] == HORIZONTAL) {
                    for (auto& p : partitions) {
                        int y1 = p.nodes.front().c.y;
                        int y2 = p.nodes.back().c.y;

                        for (int i = p.v1; i < p.nodes.size() - p.v2; i++) {
                            int R1 = p.nodes[i].c.y - y1; // y1 <- node
                            int R2 = y2 - p.nodes[i].c.y; // node -> y2

                            p.nodes[i].R1 = R1;
                            p.nodes[i].R2 = R2;

                            int R = R1 + R2;
                            
                            float I1;
                            float I2;

                            if (p.v1 && !p.v2) {
                                I1 = *(p.nodes[i].current_source);
                                I2 = 0.0;
                            }
                            else if (!p.v1 && p.v2) {
                                I1 = 0.0;
                                I2 = *(p.nodes[i].current_source);
                            }
                            else {
                                I1 = *(p.nodes[i].current_source) * R2 / R; // y1 <- node
                                I2 = *(p.nodes[i].current_source) * R1 / R; // node -> y2
                            }

                            for (int j = 0; j <= i; j++) {
                                p.nodes[j].current += I1;
                            }
                            for (int j = i + 1; j < p.nodes.size(); j++) {
                                p.nodes[j].current -= I2;
                            }

                        }
                        if (p.v1) {
                            *(p.nodes.front().current_source) += abs(p.nodes.front().current);
                        }

                        if (p.v2) {
                            *(p.nodes.back().current_source) += abs(p.nodes.back().current);
                        }
                    }
                }
            }
        }

        void get_ir_drop() {
            for (int metal = metal_layer.size() - 1; metal >= 0; metal--) {
                vector<local_partition>& partitions = partition_map[metal];

                if (orientation[metal] == VERTICAL) {
                    for (auto& p : partitions) {
                        if (p.v1 && p.v2) {
                            vector<float> *a = p.nodes.front().ir_drop_ptr;
                            vector<float> *b = p.nodes.back().ir_drop_ptr;

                            for (int i = 1; i < p.nodes.size() - 1; i++) {
                                (*p.nodes[i].ir_drop_ptr)[metal * 2] = (*p.nodes[i - 1].ir_drop_ptr)[metal * 2] + p.nodes[i - 1].current * (p.nodes[i].c.x - p.nodes[i - 1].c.x) * p.resistance_per_unit;
                                if (metal != 0)
                                    (*p.nodes[i].ir_drop_ptr)[metal * 2 - 1] = (*p.nodes[i].current_source) * p.nodes[i].resistance;

                                float W = (float)p.nodes[i].R1 / (p.nodes[i].R1 + p.nodes[i].R2);

                                for (int m = metal_layer.size() - 1; m > metal; m--) {
                                    (*p.nodes[i].ir_drop_ptr)[m * 2]     = (*a)[m * 2]     + W * ((*b)[m * 2]     - (*a)[m * 2]    );
                                    (*p.nodes[i].ir_drop_ptr)[m * 2 - 1] = (*a)[m * 2 - 1] + W * ((*b)[m * 2 - 1] - (*a)[m * 2 - 1]);
                                }

                            }
                        }
                        else if (p.v1) {
                            vector<float> *a = p.nodes.front().ir_drop_ptr;

                            for (int i = 1; i < p.nodes.size(); i++) {
                                (*p.nodes[i].ir_drop_ptr)[metal * 2] = (*p.nodes[i - 1].ir_drop_ptr)[metal * 2] + p.nodes[i - 1].current * (p.nodes[i].c.x - p.nodes[i - 1].c.x) * p.resistance_per_unit;
                                if (metal != 0)
                                    (*p.nodes[i].ir_drop_ptr)[metal * 2 - 1] = (*p.nodes[i].current_source) * p.nodes[i].resistance;

                                for (int m = metal_layer.size() - 1; m > metal; m--) {
                                    (*p.nodes[i].ir_drop_ptr)[m * 2]     = (*a)[m * 2]    ;
                                    (*p.nodes[i].ir_drop_ptr)[m * 2 - 1] = (*a)[m * 2 - 1];
                                }
                            }
                        }
                        else if (p.v2) {
                            vector<float> *b = p.nodes.back().ir_drop_ptr;

                            for (int i = p.nodes.size() - 2; i >= 0; i--) {
                                (*p.nodes[i].ir_drop_ptr)[metal * 2] = (*p.nodes[i + 1].ir_drop_ptr)[metal * 2] - p.nodes[i + 1].current * (p.nodes[i + 1].c.x - p.nodes[i].c.x) * p.resistance_per_unit;
                                if (metal != 0)
                                    (*p.nodes[i].ir_drop_ptr)[metal * 2 - 1] = (*p.nodes[i].current_source) * p.nodes[i].resistance;

                                for (int m = metal_layer.size() - 1; m > metal; m--) {
                                    (*p.nodes[i].ir_drop_ptr)[m * 2]     = (*b)[m * 2]    ;
                                    (*p.nodes[i].ir_drop_ptr)[m * 2 - 1] = (*b)[m * 2 - 1];
                                }
                            }
                        }
                    }
                }
                else if (orientation[metal] == HORIZONTAL) {
                    for (auto& p : partitions) {
                        if (p.v1 && p.v2) {
                            vector<float> *a = p.nodes.front().ir_drop_ptr;
                            vector<float> *b = p.nodes.back().ir_drop_ptr;

                            for (int i = 1; i < p.nodes.size() - 1; i++) {
                                (*p.nodes[i].ir_drop_ptr)[metal * 2] = (*p.nodes[i - 1].ir_drop_ptr)[metal * 2] + p.nodes[i - 1].current * (p.nodes[i].c.y - p.nodes[i - 1].c.y) * p.resistance_per_unit;
                                if (metal != 0)
                                    (*p.nodes[i].ir_drop_ptr)[metal * 2 - 1] = (*p.nodes[i].current_source) * p.nodes[i].resistance;

                                float W = (float)p.nodes[i].R1 / (p.nodes[i].R1 + p.nodes[i].R2);

                                for (int m = metal_layer.size() - 1; m > metal; m--) {
                                    (*p.nodes[i].ir_drop_ptr)[m * 2]     = (*a)[m * 2]     + W * ((*b)[m * 2]     - (*a)[m * 2]    );
                                    (*p.nodes[i].ir_drop_ptr)[m * 2 - 1] = (*a)[m * 2 - 1] + W * ((*b)[m * 2 - 1] - (*a)[m * 2 - 1]);
                                }
                            }
                        }
                        else if (p.v1) {
                            vector<float> *a = p.nodes.front().ir_drop_ptr;

                            for (int i = 1; i < p.nodes.size(); i++) {
                                (*p.nodes[i].ir_drop_ptr)[metal * 2] = (*p.nodes[i - 1].ir_drop_ptr)[metal * 2] + p.nodes[i - 1].current * (p.nodes[i].c.y - p.nodes[i - 1].c.y) * p.resistance_per_unit;
                                if (metal != 0)
                                    (*p.nodes[i].ir_drop_ptr)[metal * 2 - 1] = (*p.nodes[i].current_source) * p.nodes[i].resistance;

                                for (int m = metal_layer.size() - 1; m > metal; m--) {
                                    (*p.nodes[i].ir_drop_ptr)[m * 2]     = (*a)[m * 2]    ;
                                    (*p.nodes[i].ir_drop_ptr)[m * 2 - 1] = (*a)[m * 2 - 1];
                                }
                            }
                        }
                        else if (p.v2) {
                            vector<float> *b = p.nodes.back().ir_drop_ptr;

                            for (int i = p.nodes.size() - 2; i >= 0; i--) {
                                (*p.nodes[i].ir_drop_ptr)[metal * 2] = (*p.nodes[i + 1].ir_drop_ptr)[metal * 2] - p.nodes[i + 1].current * (p.nodes[i + 1].c.x - p.nodes[i].c.x) * p.resistance_per_unit;
                                if (metal != 0)
                                    (*p.nodes[i].ir_drop_ptr)[metal * 2 - 1] = (*p.nodes[i].current_source) * p.nodes[i].resistance;

                                for (int m = metal_layer.size() - 1; m > metal; m--) {
                                    (*p.nodes[i].ir_drop_ptr)[m * 2]     = (*b)[m * 2]    ;
                                    (*p.nodes[i].ir_drop_ptr)[m * 2 - 1] = (*b)[m * 2 - 1];
                                }
                            }
                        }
                    }
                }
            }
        }

        void print_nodes() {
            for(int i = 0; i < current_map.size(); i++) {
                cout << metal_layer.right.at(i) << " : \n";
                for (auto& n : current_map[i]) {
                    cout << "(" << metal_layer.right.at(i) << ", " << n.c.x << ", " << n.c.y << ", " << n.current << ", "  << *n.current_source << ") \n";
                }
                cout << endl;
            }
            for(int i = 0; i < voltage_map.size(); i++) {
                cout << metal_layer.right.at(i) << " : \n";
                for (auto& n : voltage_map[i]) {
                    cout << "(" << metal_layer.right.at(i) << ", " << n.c.x << ", " << n.c.y << ", " << n.current << ", "  << *n.current_source << ") \n";
                }
                cout << endl;
            }
        }

        void print_wires() {
            for(int i = 0; i < wire_map.size(); i++) {
                cout << metal_layer.right.at(i) << " : \n";
                for (auto& w : wire_map[i]) {
                    cout << "(" << metal_layer.right.at(i) << ", " << w.c1.x << ", " << w.c1.y << ") -> (" << metal_layer.right.at(i) << ", " << w.c2.x << ", " << w.c2.y << ") " << ", " << w.resistance << ", " << w.resistance_per_unit << ") \n";
                }
                cout << endl;
            }
        }

        void print_partitions() {
            for(int i = 0; i < partition_map.size(); i++) {
                cout << metal_layer.right.at(i) << " : \n";
                for (auto& p : partition_map[i]) {
                    cout << "start" << endl;
                    cout << p.v1 << " " << p.v2 << " " << p.resistance_per_unit << endl;
                    for (auto& n : p.nodes) {
                        cout << "(" << metal_layer.right.at(i) << ", " << n.c.x << ", " << n.c.y << ", " << n.current << ", "  << *n.current_source << ") \n";
                        cout << n.ir_drop_ptr->at(0) << " " << n.ir_drop_ptr->at(1) << " " << n.ir_drop_ptr->at(2) << " " << n.ir_drop_ptr->at(3) << " " << n.ir_drop_ptr->at(4) << " " << n.ir_drop_ptr->at(5) << " " << n.ir_drop_ptr->at(6) << " " << n.ir_drop_ptr->at(7) << " " << n.ir_drop_ptr->at(8) << endl;
                    }
                    cout << "end" << endl;
                }
                cout << endl;
            }
        }

        vector<vector<node>> get_ir_map_data() {
            vector<vector<node>> out;
            vector<node> col;
            int pre_y = partition_map[0].front().nodes.front().c.y;

            for (auto& p : partition_map[0]) {
                if (pre_y != p.nodes.front().c.y) {
                    if (col.size() > 0)
                        out.push_back(col);
                    col.clear();

                    pre_y = p.nodes.front().c.y;
                }

                for (int i = 0; i < p.nodes.size(); i++) {
                    col.push_back(p.nodes[i]);
                }
            }

            return out;
        }

        boost::multi_array<float, 3> get_ir_map(vector<vector<node>> chip) {
            float h_gap = (float)x_max / H;
            float w_gap = (float)y_max / W;

            vector<vector<vector<float>>> value_map;
            vector<float> y_map;

            // vertical interpolation
            for(auto &c : chip) {
                vector<vector<float>> values;

                int index = 0;
                float w, v1, v2;
                float x1, x2;

                for(int h = 0; h < H; h++) {
                    vector<float> value (metal_layer.size() + via_layer.size(), 0.0);

                    x1 = h * h_gap;
                    x2 = x1 + h_gap;

                    if(index == c.size() - 1) {
                        if (x1 <= c[index].c.x && c[index].c.x <= x2) {
                            for (int m = 0; m < metal_layer.size() + via_layer.size(); m++) {
                                w = ((*c[index].ir_drop_ptr)[m] - (*c[index - 1].ir_drop_ptr)[m]) / (c[index].c.x - c[index - 1].c.x);
                                v1 = (x1 - c[index - 1].c.x) * w + (*c[index - 1].ir_drop_ptr)[m];
                                value[m] += (v1 + (*c[index].ir_drop_ptr)[m]) * (c[index].c.x - x1) / 2;

                                value[m] += (*c[index].ir_drop_ptr)[m] * (x2 - c[index].c.x);
                                value[m] /= h_gap;
                            }
                        }
                        else if(x2 < c[index].c.x) {
                            for (int m = 0; m < metal_layer.size() + via_layer.size(); m++) {
                                w = ((*c[index].ir_drop_ptr)[m] - (*c[index - 1].ir_drop_ptr)[m]) / (c[index].c.x - c[index - 1].c.x);
                                v1 = (x1 - c[index - 1].c.x) * w + (*c[index - 1].ir_drop_ptr)[m];
                                v2 = (x2 - c[index - 1].c.x) * w + (*c[index - 1].ir_drop_ptr)[m];
                                value[m] = (v1 + v2) / 2;
                            }
                        }
                        else {
                            for (int m = 0; m < metal_layer.size() + via_layer.size(); m++)
                                value[m] = (*c[index].ir_drop_ptr)[m];

                        }
                        values.push_back(value);
                        continue;
                    }

                    if(index == 0) {
                        if (x1 <= c[index].c.x && c[index].c.x <= x2) {
                            for (int m = 0; m < metal_layer.size() + via_layer.size(); m++)
                                value[m] += (*c[index].ir_drop_ptr)[m] * (c[index].c.x - x1);
                        }
                        else {
                            for (int m = 0; m < metal_layer.size() + via_layer.size(); m++)
                                value[m] = (*c[index].ir_drop_ptr)[m];

                            values.push_back(value);
                            continue;
                        }
                    }
                    else {
                        if (x1 <= c[index].c.x && c[index].c.x <= x2) {
                            for (int m = 0; m < metal_layer.size() + via_layer.size(); m++) {
                                w = ((*c[index].ir_drop_ptr)[m] - (*c[index - 1].ir_drop_ptr)[m]) / (c[index].c.x - c[index - 1].c.x);
                                v1 = (x1 - c[index - 1].c.x) * w + (*c[index - 1].ir_drop_ptr)[m];
                                value[m] += (v1 + (*c[index].ir_drop_ptr)[m]) * (c[index].c.x - x1) / 2;
                            }
                        }
                        else {
                            for (int m = 0; m < metal_layer.size() + via_layer.size(); m++) {
                                w = ((*c[index].ir_drop_ptr)[m] - (*c[index - 1].ir_drop_ptr)[m]) / (c[index].c.x - c[index - 1].c.x);
                                v1 = (x1 - c[index - 1].c.x) * w + (*c[index - 1].ir_drop_ptr)[m];
                                v2 = (x2 - c[index - 1].c.x) * w + (*c[index - 1].ir_drop_ptr)[m];
                                value[m] = (v1 + v2) / 2;
                            }

                            values.push_back(value);
                            continue;
                        }
                    }

                    while(x1 <= c[index].c.x && c[index + 1].c.x <= x2) {
                        for (int m = 0; m < metal_layer.size() + via_layer.size(); m++)
                            value[m] += (c[index + 1].c.x - c[index].c.x) * ((*c[index + 1].ir_drop_ptr)[m] + (*c[index].ir_drop_ptr)[m]) / 2;
                        index++;
                        if(index == c.size() - 1)
                            break;
                    }

                    if(index == c.size() - 1) {
                        for (int m = 0; m < metal_layer.size() + via_layer.size(); m++)
                            value[m] += (*c[index].ir_drop_ptr)[m] * (x2 - c[index].c.x);
                    }
                    else {
                        for (int m = 0; m < metal_layer.size() + via_layer.size(); m++) {
                            w = ((*c[index + 1].ir_drop_ptr)[m] - (*c[index].ir_drop_ptr)[m]) / (c[index + 1].c.x - c[index].c.x);
                            v2 = (x2 - c[index].c.x) * w +(*c[index].ir_drop_ptr)[m];
                            value[m] += ((*c[index].ir_drop_ptr)[m] + v2) * (x2 - c[index].c.x) / 2;
                        }
                    }
                    for (int m = 0; m < metal_layer.size() + via_layer.size(); m++)
                        value[m] /= h_gap;

                    values.push_back(value);

                    if(index != c.size() - 1)
                        index++;
                }

                value_map.push_back(values);
                y_map.push_back(c.front().c.y);
            }

            // horizontal interpolation
            boost::multi_array<float, 3> out((boost::extents[metal_layer.size() + via_layer.size()][H][W]));

            int index = 0;
            float s, v1, v2;
            float y1, y2;

            for(int w = 0; w < W; w++) {
                y1 = w * w_gap;
                y2 = y1 + w_gap;
                
                if(index == y_map.size() - 1) {
                    if (y1 <= y_map[index] && y_map[index] <= y2) {
                        for (int h = 0; h < H; h++)
                            for (int m = 0; m < metal_layer.size() + via_layer.size(); m++) {
                                s = (value_map[index][h][m] - value_map[index - 1][h][m]) / (y_map[index] - y_map[index - 1]);
                                v1 = (y1 - y_map[index - 1]) * s + value_map[index - 1][h][m];
                                out[m][h][w] += (v1 + value_map[index][h][m]) * (y_map[index] - y1) / 2;
                                out[m][h][w] += value_map[index][h][m] * (y2 - y_map[index]);
                                out[m][h][w] /= w_gap;
                            }
                    }
                    else if(y2 < y_map[index]) {
                        for (int h = 0; h < H; h++)
                            for (int m = 0; m < metal_layer.size() + via_layer.size(); m++){
                                s = (value_map[index][h][m] - value_map[index - 1][h][m]) / (y_map[index] - y_map[index - 1]);
                                v1 = (y1 - y_map[index - 1]) * s + value_map[index - 1][h][m];
                                v2 = (y2 - y_map[index - 1]) * s + value_map[index - 1][h][m];
                                out[m][h][w] = (v1 + v2) / 2;
                            }
                    }
                    else {
                        for (int h = 0; h < H; h++)
                            for (int m = 0; m < metal_layer.size() + via_layer.size(); m++)
                                out[m][h][w] = value_map[index][h][m];

                    }

                    continue;
                }

                if(index == 0) {
                    if (y1 <= y_map[index] && y_map[index] <= y2) {
                        for (int h = 0; h < H; h++)
                            for (int m = 0; m < metal_layer.size() + via_layer.size(); m++)
                                out[m][h][w] += value_map[index][h][m] * (y_map[index] - y1);
                    }
                    else {
                        for (int h = 0; h < H; h++)
                            for (int m = 0; m < metal_layer.size() + via_layer.size(); m++)
                                out[m][h][w] = value_map[index][h][m];

                        continue;
                    }
                }
                else {
                    if (y1 <= y_map[index] && y_map[index] <= y2) {
                        for (int h = 0; h < H; h++)
                            for (int m = 0; m < metal_layer.size() + via_layer.size(); m++) {
                                s = (value_map[index][h][m] - value_map[index - 1][h][m]) / (y_map[index] - y_map[index - 1]);
                                v1 = (y1 - y_map[index - 1]) * s + value_map[index - 1][h][m];
                                out[m][h][w] += (v1 + value_map[index][h][m]) * (y_map[index] - y1) / 2;
                            }
                    }
                    else {
                        for (int h = 0; h < H; h++)
                            for (int m = 0; m < metal_layer.size() + via_layer.size(); m++) {
                                s = (value_map[index][h][m] - value_map[index - 1][h][m]) / (y_map[index] - y_map[index - 1]);
                                v1 = (y1 - y_map[index - 1]) * s + value_map[index - 1][h][m];
                                v2 = (y2 - y_map[index - 1]) * s + value_map[index - 1][h][m];
                                out[m][h][w] = (v1 + v2) / 2;
                            }

                        continue;
                    }
                }

                while(y1 <= y_map[index] && y_map[index + 1] <= y2) {
                    for (int h = 0; h < H; h++)
                        for (int m = 0; m < metal_layer.size() + via_layer.size(); m++)
                            out[m][h][w] += (value_map[index + 1][h][m] + value_map[index][h][m]) * (y_map[index + 1] - y_map[index]) / 2;
                    index++;
                    if(index == y_map.size() - 1)
                        break;
                }

                if(index == y_map.size() - 1) {
                    for (int h = 0; h < H; h++)
                        for (int m = 0; m < metal_layer.size() + via_layer.size(); m++)
                            out[m][h][w] += value_map[index][h][m] * (y2 - y_map[index]);
                }
                else {
                    for (int h = 0; h < H; h++)
                        for (int m = 0; m < metal_layer.size() + via_layer.size(); m++) {
                            s = (value_map[index + 1][h][m] - value_map[index][h][m]) / (y_map[index + 1] - y_map[index]);
                            v2 = (y2 - y_map[index]) * s + value_map[index][h][m];
                            out[m][h][w] += (value_map[index][h][m] + v2) * (y2 - y_map[index]) / 2;
                        }
                }

                for (int h = 0; h < H; h++)
                    for (int m = 0; m < metal_layer.size() + via_layer.size(); m++)
                        out[m][h][w] /= w_gap;
                
                if(index != y_map.size() - 1)
                    index++;
            }

            return out;
        }

        boost::multi_array<float, 3> get_distance() {
            boost::multi_array<float, 3> out((boost::extents[metal_layer.size() * 2 - 3][H][W]));

            float h_gap = (float)x_max / H;
            float w_gap = (float)y_max / W;

            int h1, h2, w1, w2;
            int wh1, wh2, ww1, ww2;

            vector<h_wire> h_wires;
            vector<h_wire> new_h_wires;
            vector<h_wire> no_dist_h_wires;
            vector<v_wire> v_wires;
            vector<v_wire> new_v_wires;
            vector<v_wire> no_dist_v_wires;

            int cnt = 0;

            for (int m = 1; m < metal_layer.size(); m++) {
                auto& wires = wire_map[m];

                if(orientation[m] == HORIZONTAL){
                    // start
                    h_wires.clear();
                    new_h_wires.clear();
                    

                    h_wire h_first;
                    h_first.h  = 0;
                    h_first.w1 = 0;
                    h_first.w2 = y_max;
                    h_wires.push_back(h_first);

                    for(int i = 0; i < wires.size(); i++) {
                        wire& wire = wires[i];

                        wh1 = wire.c1.x;
                        ww1 = wire.c1.y;
                        ww2 = wire.c2.y;


                        for(auto& h_wire : h_wires) {
                            h1 = h_wire.h;
                            w1 = h_wire.w1;
                            w2 = h_wire.w2;
                            
                            if(w2 < ww1 || ww2 < w1) { // no overlap
                                new_h_wires.push_back({h1, w1, w2});
                            }
                            else if(ww1 <= w1 && w2 <= ww2) { // full overlap
                                for (int h = h1/h_gap + 0.5; h < floor(wh1/h_gap + 0.5); h++) {
                                    float distance = h * h_gap - h1;
                                    for (int w = w1/w_gap + 0.5; w < floor(w2/w_gap + 0.5); w++) {
                                        out[cnt][h][w] = distance;
                                    }
                                }
                            }
                            else if(w1 < ww1 && ww2 < w2) { // mid overlap
                                for (int h = h1/h_gap + 0.5; h < floor(wh1/h_gap + 0.5); h++) {
                                    float distance = h * h_gap - h1;
                                    for (int w = ww1/w_gap + 0.5; w < floor(ww2/w_gap + 0.5); w++) {
                                        out[cnt][h][w] = distance;
                                    }
                                }
                                new_h_wires.push_back({h1, w1, ww1});
                                new_h_wires.push_back({h1, ww2, w2});
                            }
                            else if(ww2 < w2) { // left overlap
                                for (int h = h1/h_gap + 0.5; h < floor(wh1/h_gap + 0.5); h++) {
                                    float distance = h * h_gap - h1;
                                    for (int w = w1/w_gap + 0.5; w < floor(ww2/w_gap + 0.5); w++) {
                                        out[cnt][h][w] = distance;
                                    }
                                }
                                new_h_wires.push_back({h1, ww2, w2});
                                // cout << "abc" << endl;
                            }
                            else if(w1 < ww1) { // right overlap
                                for (int h = h1/h_gap + 0.5; h < floor(wh1/h_gap + 0.5); h++) {
                                    float distance = h * h_gap - h1;
                                    for (int w = ww1/w_gap + 0.5; w < floor(w2/w_gap + 0.5); w++) {
                                        out[cnt][h][w] = distance;
                                    }
                                }
                                new_h_wires.push_back({h1, w1, ww1});
                            }
                        }
                        h_wires = new_h_wires;
                        new_h_wires.clear();
                    }
                    no_dist_h_wires = h_wires;
                    
                    // iteration
                    for (int index = 0; index < wires.size(); index++) {
                        h_wires.clear();
                        new_h_wires.clear();

                        h_wire h_first;
                        h_first.h  = wires[index].c1.x;
                        h_first.w1 = wires[index].c1.y;
                        h_first.w2 = wires[index].c2.y;
                        h_wires.push_back(h_first);

                        for(int i = index + 1; i < wires.size(); i++) {
                            wire& wire = wires[i];

                            wh1 = wire.c1.x;
                            ww1 = wire.c1.y;
                            ww2 = wire.c2.y;

                            for(auto& h_wire : h_wires) {
                                h1 = h_wire.h;
                                w1 = h_wire.w1;
                                w2 = h_wire.w2;

                                if(w2 < ww1 || ww2 < w1) { // no overlap
                                    new_h_wires.push_back({h1, w1, w2});
                                }
                                else if(ww1 <= w1 && w2 <= ww2) { // full overlap
                                    for (int h = h1/h_gap + 0.5; h < floor(wh1/h_gap + 0.5); h++) {
                                        float distance = min((h * h_gap - h1), (wh1 - h * h_gap));
                                        for (int w = w1/w_gap + 0.5; w < floor(w2/w_gap + 0.5); w++) {
                                            out[cnt][h][w] = distance;
                                        }
                                    }
                                }
                                else if(w1 < ww1 && ww2 < w2) { // mid overlap
                                    for (int h = h1/h_gap + 0.5; h < floor(wh1/h_gap + 0.5); h++) {
                                        float distance = min((h * h_gap - h1), (wh1 - h * h_gap));
                                        for (int w = ww1/w_gap + 0.5; w < floor(ww2/w_gap + 0.5); w++) {
                                            out[cnt][h][w] = distance;
                                        }
                                    }
                                    new_h_wires.push_back({h1, w1, ww1});
                                    new_h_wires.push_back({h1, ww2, w2});
                                }
                                else if(ww2 < w2) { // left overlap
                                    for (int h = h1/h_gap + 0.5; h < floor(wh1/h_gap + 0.5); h++) {
                                        float distance = min((h * h_gap - h1), (wh1 - h * h_gap));
                                        for (int w = w1/w_gap + 0.5; w < floor(ww2/w_gap + 0.5); w++) {
                                            out[cnt][h][w] = distance;
                                        }
                                    }
                                    new_h_wires.push_back({h1, ww2, w2});
                                }
                                else if(w1 < ww1) { // right overlap
                                    for (int h = h1/h_gap + 0.5; h < floor(wh1/h_gap + 0.5); h++) {
                                        float distance = min((h * h_gap - h1), (wh1 - h * h_gap));
                                        for (int w = ww1/w_gap + 0.5; w < floor(w2/w_gap + 0.5); w++) {
                                            out[cnt][h][w] = distance;
                                        }
                                    }
                                    new_h_wires.push_back({h1, w1, ww1});
                                }
                            }
                            h_wires = new_h_wires;
                            new_h_wires.clear();
                        }
                        for(auto& h_wire : h_wires) { // no overlap wires
                            h1 = h_wire.h;
                            w1 = h_wire.w1;
                            w2 = h_wire.w2;

                            for (int h = h1/h_gap + 0.5; h < H; h++) {
                                float distance = h * h_gap - h1;
                                for (int w = w1/w_gap + 0.5; w < floor(w2/w_gap + 0.5); w++) {
                                    out[cnt][h][w] = distance;
                                }
                            }
                        }
                    }

                    // interation no distance data
                    for(auto& h_wire : no_dist_h_wires) {
                        w1 = h_wire.w1;
                        w2 = h_wire.w2;

                        for (int h = 0; h < H; h++) {
                            float distance;

                            if(int(w1/w_gap + 0.5) == 0) {
                                distance = out[cnt][h][w2/w_gap + 0.5];
                            }
                            else {
                                distance = out[cnt][h][w1/w_gap + 0.5 - 1];
                            }
                            
                            for (int w = w1/w_gap + 0.5; w < floor(w2/w_gap + 0.5); w++) {
                                out[cnt][h][w] = distance;
                            }
                        }
                    }
                }
                else if(orientation[m] == VERTICAL){
                    // start
                    v_wires.clear();
                    new_v_wires.clear();

                    v_wire v_first;
                    v_first.w  = 0;
                    v_first.h1 = 0;
                    v_first.h2 = x_max;
                    v_wires.push_back(v_first);

                    for(int i = 0; i < wires.size(); i++) {
                        wire& wire = wires[i];

                        ww1 = wire.c1.y;
                        wh1 = wire.c1.x;
                        wh2 = wire.c2.x;

                        for(auto& v_wire : v_wires) {
                            w1 = v_wire.w;
                            h1 = v_wire.h1;
                            h2 = v_wire.h2;

                            if(h2 < wh1 || wh2 < h1) { // no overlap
                                new_v_wires.push_back({w1, h1, h2});
                            }
                            else if(wh1 <= h1 && h2 <= wh2) { // full overlap
                                for (int w = w1/w_gap + 0.5; w < floor(ww1/w_gap + 0.5); w++) {
                                    float distance = (w + 0.5) * w_gap - w1;
                                    for (int h = h1/h_gap + 0.5; h < floor(h2/h_gap + 0.5); h++) {
                                        out[cnt][h][w] = distance;
                                    }
                                }
                            }
                            else if(h1 < wh1 && wh2 < h2) { // mid overlap
                                for (int w = w1/w_gap + 0.5; w < floor(ww1/w_gap + 0.5); w++) {
                                    float distance = (w + 0.5) * w_gap - w1;
                                    for (int h = h1/h_gap; h < floor(h2/h_gap + 0.5); h++) {
                                        out[cnt][h][w] = distance;
                                    }
                                }
                                new_v_wires.push_back({w1, h1, wh1});
                                new_v_wires.push_back({w1, wh2, h2});
                            }
                            else if(wh2 < h2) { // left overlap
                                for (int w = w1/w_gap + 0.5; w < floor(ww1/w_gap + 0.5); w++) {
                                    float distance = (w + 0.5) * w_gap - w1;
                                    for (int h = h1/h_gap; h < floor(wh2/h_gap + 0.5); h++) {
                                        out[cnt][h][w] = distance;
                                    }
                                }
                                new_v_wires.push_back({w1, wh2, h2});
                            }
                            else if(h1 < wh1) { // right overlap
                                for (int w = w1/w_gap + 0.5; w < floor(ww1/w_gap + 0.5); w++) {
                                    float distance = (w + 0.5) * w_gap - w1;
                                    for (int h = wh1/h_gap + 0.5; h < floor(h2/h_gap + 0.5); h++) {
                                        out[cnt][h][w] = distance;
                                    }
                                }
                                new_v_wires.push_back({w1, h1, wh1});
                            }
                        }
                        v_wires = new_v_wires;
                        new_v_wires.clear();
                    }
                    no_dist_v_wires = v_wires;
                    
                    // iteration
                    for (int index = 0; index < wires.size(); index++) {
                        v_wires.clear();
                        new_v_wires.clear();

                        v_wire v_first;
                        v_first.w  = wires[index].c1.y;
                        v_first.h1 = wires[index].c1.x;
                        v_first.h2 = wires[index].c2.x;
                        v_wires.push_back(v_first);

                        for(int i = index + 1; i < wires.size(); i++) {
                            wire& wire = wires[i];

                            ww1 = wire.c1.y;
                            wh1 = wire.c1.x;
                            wh2 = wire.c2.x;

                            for(auto& v_wire : v_wires) {
                                w1 = v_wire.w;
                                h1 = v_wire.h1;
                                h2 = v_wire.h2;

                                if(h2 < wh1 || wh2 < h1) { // no overlap
                                    new_v_wires.push_back({w1, h1, h2});
                                }
                                else if(wh1 <= h1 && h2 <= wh2) { // full overlap
                                    for (int w = w1/w_gap + 0.5; w < floor(ww1/w_gap + 0.5); w++) {
                                        float distance = min(((w + 0.5) * w_gap - w1), (ww1 - (w + 0.5) * w_gap));
                                        for (int h = h1/h_gap + 0.5; h < floor(h2/h_gap + 0.5); h++) {
                                            out[cnt][h][w] = distance;
                                        }
                                    }
                                }
                                else if(h1 < wh1 && wh2 < h2) { // mid overlap
                                    for (int w = w1/w_gap + 0.5; w < floor(ww1/w_gap + 0.5); w++) {
                                        float distance = min(((w + 0.5) * w_gap - w1), (ww1 - (w + 0.5) * w_gap));
                                        for (int h = h1/h_gap + 0.5; h < floor(h2/h_gap + 0.5); h++) {
                                            out[cnt][h][w] = distance;
                                        }
                                    }
                                    new_v_wires.push_back({w1, h1, wh1});
                                    new_v_wires.push_back({w1, wh2, h2});
                                }
                                else if(wh2 < h2) { // left overlap
                                    for (int w = w1/w_gap + 0.5; w < floor(ww1/w_gap + 0.5); w++) {
                                        float distance = min(((w + 0.5) * w_gap - w1), (ww1 - (w + 0.5) * w_gap));
                                        for (int h = h1/h_gap + 0.5; h < floor(wh2/h_gap + 0.5); h++) {
                                            out[cnt][h][w] = distance;
                                        }
                                    }
                                    new_v_wires.push_back({w1, wh2, h2});
                                }
                                else if(h1 < wh1) { // right overlap
                                    for (int w = w1/w_gap + 0.5; w < floor(ww1/w_gap + 0.5); w++) {
                                        float distance = min(((w + 0.5) * w_gap - w1), (ww1 - (w + 0.5) * w_gap));
                                        for (int h = wh1/h_gap + 0.5; h < floor(h2/h_gap + 0.5); h++) {
                                            out[cnt][h][w] = distance;
                                        }
                                    }
                                    new_v_wires.push_back({w1, h1, wh1});
                                }
                            }
                            v_wires = new_v_wires;
                            new_v_wires.clear();
                        }
                        for(auto& v_wire : v_wires) { // no overlap wires
                            w1 = v_wire.w;
                            h1 = v_wire.h1;
                            h2 = v_wire.h2;

                            for (int w = w1/w_gap + 0.5; w < W; w++) {
                                float distance = (w + 0.5) * w_gap - w1;
                                for (int h = h1/h_gap + 0.5; h < floor(h2/h_gap + 0.5); h++) {
                                    out[cnt][h][w] = distance;
                                }
                            }
                        }

                        // interation no distance data
                        for(auto& v_wire : no_dist_v_wires) {
                            h1 = v_wire.h1;
                            h2 = v_wire.h2;

                            for (int w = 0; w < W; w++) {
                                float distance;

                                if(int(h1/h_gap + 0.5) == 0) {
                                    distance = out[cnt][h2/h_gap + 0.5][w];
                                }
                                else {
                                    distance = out[cnt][h1/h_gap + 0.5 - 1][w];
                                }
                                
                                for (int h = h1/h_gap + 0.5; h < floor(h2/h_gap + 0.5); h++) {
                                    out[cnt][h][w] = distance;
                                }
                            }
                        }
                    }
                }
                
                cnt++;
            }

            for (int m = 0; m < metal_layer.size() - 2; m++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        out[cnt][h][w] = out[m][h][w] + out[m+1][h][w];
                    }
                }

                cnt++;
            }

            return out;
        }

        boost::multi_array<float, 3> get_resistance() {
            boost::multi_array<float, 3> out(boost::extents[2 * metal_layer.size() - 3][H][W]);

            float h_gap = (float)x_max / H;
            float w_gap = (float)y_max / W;

            int cnt = 0;
            for (int m = 1; m < metal_layer.size(); m++) {
                auto& wires = wire_map[m];

                for (auto &w : wires) {
                    if(w.c1.x == w.c2.x) {
                        int x1 = floor(w.c2.x / h_gap - 0.5);
                        int x2 = x1 + 1;
                        float w1 = x2 - (w.c2.x / h_gap - 0.5);
                        float w2 = (w.c2.x / h_gap - 0.5) - x1;

                        for(int i = floor(w.c1.y/w_gap - 0.5); i <= ceil(w.c2.y/w_gap - 0.5); i++) {
                            if(i >= 0 && i < W && x1 >= 0 && x1 < H)
                                out[cnt][x1][i] = w1;
                            if(i >= 0 && i < W && x2 >= 0 && x2 < H)
                                out[cnt][x2][i] = w2;
                        }
                    }
                    else {
                        int y1 = floor(w.c2.y / w_gap - 0.5);
                        int y2 = y1 + 1;
                        float w1 = y2 - (w.c2.y / w_gap - 0.5);
                        float w2 = (w.c2.y / w_gap - 0.5) - y1;

                        for(int i = floor(w.c1.x/h_gap - 0.5); i <= ceil(w.c2.x/h_gap - 0.5); i++) {
                            if(i >= 0 && i < H && y1 >= 0 && y1 < W)
                                out[cnt][i][y1] = w1;
                            if(i >= 0 && i < H && y2 >= 0 && y2 < W)
                                out[cnt][i][y2] = w2;
                        }
                    }
                }

                cnt++;
            }

            for (int m = 0; m < metal_layer.size() - 2; m++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        out[cnt][h][w] = out[m][h][w] * out[m+1][h][w];
                    }
                }
                cnt++;
            }

            return out;
        }
};


namespace py = boost::python;
namespace np = boost::python::numpy;

np::ndarray vector_to_ndarray(boost::multi_array<float, 3> marr) {
    auto array = np::from_data(
        marr.data(),
        np::dtype::get_builtin<float>(),
        py::make_tuple(
            marr.shape()[0],
            marr.shape()[1],
            marr.shape()[2]
        ),
        py::make_tuple(
            marr.strides()[0]*sizeof(float),
            marr.strides()[1]*sizeof(float),
            marr.strides()[2]*sizeof(float)
        ), 
        py::object()
    );
    np::ndarray output_array = array.copy();
    
    return output_array;
}

np::ndarray process(string netlist, int H, int W) {
    circuit c;
    c.set_orientation(
        {"m1", "m4", "m7", "m8", "m9"}, 
        {VERTICAL, HORIZONTAL, VERTICAL, HORIZONTAL, VERTICAL}
    );
    c.set_size(H, W);
    c.read_data(netlist);

    c.merge_wires();
    c.merge_nodes();
    c.get_partition();
    c.get_currents();
    c.get_ir_drop();

    auto ir_nodes = c.get_ir_map_data();
    auto ir_map = c.get_ir_map(ir_nodes);
    auto r_dist = c.get_distance();
    auto r_res = c.get_resistance();

    ir_map.resize(boost::extents[ir_map.shape()[0] + r_dist.shape()[0] + r_res.shape()[0]][H][W]);
    copy(r_dist.begin(), r_dist.end(), ir_map.end() - r_dist.size() - r_res.size());
    copy(r_res.begin(), r_res.end(), ir_map.end() - r_res.size());

    return vector_to_ndarray(ir_map);
}

BOOST_PYTHON_MODULE(libdata) {
    Py_Initialize();
    np::initialize();
    py::def("process", process);
}