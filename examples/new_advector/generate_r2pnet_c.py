#!/usr/bin/env python3

# Generates the r2pnet.h file for a given Pytorch model

import torch
import numpy as np
import os

torch.set_default_dtype(torch.float64)
# Use a very large linewidth so NumPy formats the inner arrays naturally
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=1900)
model = torch.jit.load('./model_r2p.pt')

def format_array_content(content, max_len=1900):
    content = content.strip()
    if len(content) <= max_len:
        return content
    lines = []
    while len(content) > 0:
        if len(content) <= max_len:
            lines.append(content)
            break
        break_pos = content.rfind(',', 0, max_len)
        if break_pos == -1:
            break_pos = max_len
        lines.append(content[:break_pos+1])
        content = "    " + content[break_pos+1:].lstrip()
    return "\n".join(lines)

file = open("r2pnet.h", "w")
print("/*! \\file r2pnet.h", file=file)
print(" * \\brief PLIC-Net File", file=file)
print(" * Provides the architecture for the neural network and the weights/biases.", file=file)
print(" * Use generate_r2pnet.py to generate this file for a given Pytorch model.", file=file)
print(" */\n", file=file)

print("#pragma once\n", file=file)
print("#include <cmath>\n#include <algorithm>\n", file=file)
print("namespace r2pnet {\n", file=file)

param_info = []
count = 0

for param in model.parameters():
    count = count + 1
    name = ""
    if count%2 != 0:
        name = "lay" + str(int(count/2)+1) + "_weight"
        param_info.append({'name': name, 'type': 'weight', 'data': param.detach().numpy()})
    else:
        name = "lay" + str(int((count-1)/2)+1) + "_bias"
        param_info.append({'name': name, 'type': 'bias', 'data': param.detach().numpy()})

for info in param_info:
    name = info['name']
    data = info['data']
    if info['type'] == 'weight':
        out_features = data.shape[0]
        in_features = data.shape[1]
        print(f"const double {name}[{out_features}][{in_features}] = {{", file=file)
        for m in range(out_features):
            row_data = data[m, :]
            values_str = np.array2string(row_data, separator=', ')[1:-1].strip()
            # Clean up newlines added by numpy to keep formatting precise
            values_str = values_str.replace('\n', '\n    ')
            formatted_values = format_array_content(values_str)
            comma = "," if m < out_features - 1 else ""
            print(f"    {{{formatted_values}}}{comma}", file=file)
        print("};\n", file=file)
    elif info['type'] == 'bias':
        total_size = data.shape[0]
        print(f"const double {name}[{total_size}] = {{", file=file)
        values_str = np.array2string(data, separator=', ')[1:-1].strip()
        values_str = values_str.replace('\n', '\n    ')
        formatted_values = format_array_content(values_str)
        print(f"    {formatted_values}", file=file)
        print("};\n", file=file)

# Generate the unrolled get_normal forward pass
print("inline void get_normals(const double* moments, double (&normal)[6]) {", file=file)
num_layers = int(count/2)

# Generate the intermediate temp arrays based on the exact size of each layer
for i in range(num_layers - 1):
    out_sz = param_info[2*i]['data'].shape[0]
    print(f"    double tmp{i+1}[{out_sz}] = {{0.0}};", file=file)
    
# Explicitly unroll the layers
for i in range(num_layers):
    layer_idx = i + 1
    w_data = param_info[2*i]['data']
    
    out_size = w_data.shape[0]
    in_size = w_data.shape[1]

    w_name = f"lay{layer_idx}_weight"
    b_name = f"lay{layer_idx}_bias"

    in_name = "moments" if i == 0 else f"tmp{i}"
    out_name = "normal" if i == num_layers - 1 else f"tmp{i+1}"

    if i > 0:
        print("", file=file)
    print(f"    for (int j = 0; j < {out_size}; ++j) {{", file=file)
    print(f"        {out_name}[j] = {b_name}[j];", file=file)
    print(f"        for (int i = 0; i < {in_size}; ++i)", file=file)
    print(f"            {out_name}[j] += {in_name}[i] * {w_name}[j][i];", file=file)
    
    # ReLU activation for all but the last layer
    if i < num_layers - 1:
        print(f"        if ({out_name}[j] < 0.0) {out_name}[j] = 0.0;", file=file)
    
    print("    }", file=file)

print("}\n", file=file)

reflect_subroutines = """
inline void reflect_moments_x(double (&moments)[189]);
inline void reflect_moments_y(double (&moments)[189]);
inline void reflect_moments_z(double (&moments)[189]);
inline void reflect_moments_xy(double (&moments)[189]);
inline void reflect_moments_yz(double (&moments)[189]);
inline void reflect_moments_xz(double (&moments)[189]);

inline void reflect_moments(double (&moments)[189], const double* center, int* direction_in, int* direction2_in) {
    double new_center[3];
    int direction = *direction_in;
    int direction2 = *direction2_in;
    new_center[0] = center[0];
    new_center[1] = center[1];
    new_center[2] = center[2];
    double temp;

    if (std::abs(new_center[0]) <= 1e-12) new_center[0] = 0;
    if (std::abs(new_center[1]) <= 1e-12) new_center[1] = 0;
    if (std::abs(new_center[2]) <= 1e-12) new_center[2] = 0;

    if (new_center[0] < 0 && new_center[1] >= 0 && new_center[2] >= 0) {
        direction = 1;
        reflect_moments_x(moments);
        new_center[0] = -new_center[0];
    } else if (new_center[0] >= 0 && new_center[1] < 0 && new_center[2] >= 0) {
        direction = 2;
        reflect_moments_y(moments);
        new_center[1] = -new_center[1];
    } else if (new_center[0] >= 0 && new_center[1] >= 0 && new_center[2] < 0) {
        direction = 3;
        reflect_moments_z(moments);
        new_center[2] = -new_center[2];
    } else if (new_center[0] < 0 && new_center[1] < 0 && new_center[2] >= 0) {
        direction = 4;
        reflect_moments_x(moments);
        reflect_moments_y(moments);
        new_center[0] = -new_center[0];
        new_center[1] = -new_center[1];
    } else if (new_center[0] < 0 && new_center[1] >= 0 && new_center[2] < 0) {
        direction = 5;
        reflect_moments_x(moments);
        reflect_moments_z(moments);
        new_center[0] = -new_center[0];
        new_center[2] = -new_center[2];
    } else if (new_center[0] >= 0 && new_center[1] < 0 && new_center[2] < 0) {
        direction = 6;
        reflect_moments_y(moments);
        reflect_moments_z(moments);
        new_center[1] = -new_center[1];
        new_center[2] = -new_center[2];
    } else if (new_center[0] < 0 && new_center[1] < 0 && new_center[2] < 0) {
        direction = 7;
        reflect_moments_x(moments);
        reflect_moments_y(moments);
        reflect_moments_z(moments);
        new_center[0] = -new_center[0];
        new_center[1] = -new_center[1];
        new_center[2] = -new_center[2];
    }

    if (std::abs(new_center[0] - new_center[1]) <= 1e-12 && (new_center[0] - new_center[2]) > 1e-12) {
        direction2 = 0;
    } else if (std::abs(new_center[1] - new_center[2]) <= 1e-12 && (new_center[0] - new_center[1]) > 1e-12) {
        direction2 = 0;
    } else if (std::abs(new_center[0] - new_center[1]) <= 1e-12 && (new_center[2] - new_center[0]) > 1e-12) {
        direction2 = 3;
        reflect_moments_xz(moments);
        temp = new_center[0];
        new_center[0] = new_center[2];
        new_center[2] = temp;
    } else if (std::abs(new_center[0] - new_center[2]) <= 1e-12 && (new_center[1] - new_center[0]) > 1e-12) {
        direction2 = 1;
        reflect_moments_xy(moments);
        temp = new_center[0];
        new_center[0] = new_center[1];
        new_center[1] = temp;
    } else if (std::abs(new_center[0] - new_center[2]) <= 1e-12 && (new_center[0] - new_center[1]) > 1e-12) {
        direction2 = 2;
        reflect_moments_yz(moments);
        temp = new_center[1];
        new_center[1] = new_center[2];
        new_center[2] = temp;
    } else if (std::abs(new_center[1] - new_center[2]) <= 1e-12 && (new_center[1] - new_center[0]) > 1e-12) {
        direction2 = 3;
        reflect_moments_xz(moments);
        temp = new_center[0];
        new_center[0] = new_center[2];
        new_center[2] = temp;
    } else if (new_center[1] > new_center[0] && new_center[0] >= new_center[2]) {
        direction2 = 1;
        reflect_moments_xy(moments);
        temp = new_center[0];
        new_center[0] = new_center[1];
        new_center[1] = temp;
    } else if (new_center[2] > new_center[1] && new_center[0] >= new_center[2]) {
        direction2 = 2;
        reflect_moments_yz(moments);
        temp = new_center[1];
        new_center[1] = new_center[2];
        new_center[2] = temp;
    } else if (new_center[2] > new_center[1] && new_center[1] >= new_center[0]) {
        direction2 = 3;
        reflect_moments_xz(moments);
        temp = new_center[0];
        new_center[0] = new_center[2];
        new_center[2] = temp;
    } else if (new_center[1] > new_center[0]) {
        direction2 = 4;
        reflect_moments_xy(moments);
        reflect_moments_yz(moments);
        temp = new_center[0];
        new_center[0] = new_center[1];
        new_center[1] = temp;
        temp = new_center[1];
        new_center[1] = new_center[2];
        new_center[2] = temp;
    } else if (new_center[2] > new_center[1]) {
        direction2 = 5;
        reflect_moments_xy(moments);
        reflect_moments_xz(moments);
        temp = new_center[0];
        new_center[0] = new_center[1];
        new_center[1] = temp;
        temp = new_center[0];
        new_center[0] = new_center[2];
        new_center[2] = temp;
    }

    *direction_in = direction;
    *direction2_in = direction2;
}

inline void reflect_moments_x(double (&moments)[189]) {
    double temp;
    for (int k = 0; k <= 2; ++k) {
        for (int j = 0; j <= 2; ++j) {
            for (int i = 0; i <= 2; ++i) {
                if (i == 0) {
                    for (int n = 0; n <= 6; ++n) {
                        if (n == 1 || n == 4) {
                            temp = moments[7*(i*9+j*3+k)+n];
                            moments[7*(i*9+j*3+k)+n] = -moments[7*(2*9+j*3+k)+n];
                            moments[7*(2*9+j*3+k)+n] = -temp;
                        } else {
                            temp = moments[7*(i*9+j*3+k)+n];
                            moments[7*(i*9+j*3+k)+n] = moments[7*(2*9+j*3+k)+n];
                            moments[7*(2*9+j*3+k)+n] = temp;
                        }
                    }
                } else if (i == 1) {
                    moments[7*(i*9+j*3+k)+1] = -moments[7*(i*9+j*3+k)+1];
                    moments[7*(i*9+j*3+k)+4] = -moments[7*(i*9+j*3+k)+4];
                }
            }
        }
    }
}

inline void reflect_moments_y(double (&moments)[189]) {
    double temp;
    for (int k = 0; k <= 2; ++k) {
        for (int j = 0; j <= 2; ++j) {
            for (int i = 0; i <= 2; ++i) {
                if (j == 0) {
                    for (int n = 0; n <= 6; ++n) {
                        if (n == 2 || n == 5) {
                            temp = moments[7*(i*9+j*3+k)+n];
                            moments[7*(i*9+j*3+k)+n] = -moments[7*(i*9+2*3+k)+n];
                            moments[7*(i*9+2*3+k)+n] = -temp;
                        } else {
                            temp = moments[7*(i*9+j*3+k)+n];
                            moments[7*(i*9+j*3+k)+n] = moments[7*(i*9+2*3+k)+n];
                            moments[7*(i*9+2*3+k)+n] = temp;
                        }
                    }
                } else if (j == 1) {
                    moments[7*(i*9+j*3+k)+2] = -moments[7*(i*9+j*3+k)+2];
                    moments[7*(i*9+j*3+k)+5] = -moments[7*(i*9+j*3+k)+5];
                }
            }
        }
    }
}

inline void reflect_moments_z(double (&moments)[189]) {
    double temp;
    for (int k = 0; k <= 2; ++k) {
        for (int j = 0; j <= 2; ++j) {
            for (int i = 0; i <= 2; ++i) {
                if (k == 0) {
                    for (int n = 0; n <= 6; ++n) {
                        if (n == 3 || n == 6) {
                            temp = moments[7*(i*9+j*3+k)+n];
                            moments[7*(i*9+j*3+k)+n] = -moments[7*(i*9+j*3+2)+n];
                            moments[7*(i*9+j*3+2)+n] = -temp;
                        } else {
                            temp = moments[7*(i*9+j*3+k)+n];
                            moments[7*(i*9+j*3+k)+n] = moments[7*(i*9+j*3+2)+n];
                            moments[7*(i*9+j*3+2)+n] = temp;
                        }
                    }
                } else if (k == 1) {
                    moments[7*(i*9+j*3+k)+3] = -moments[7*(i*9+j*3+k)+3];
                    moments[7*(i*9+j*3+k)+6] = -moments[7*(i*9+j*3+k)+6];
                }
            }
        }
    }
}

inline void reflect_moments_xy(double (&moments)[189]) {
    double temp;
    for (int k = 0; k <= 2; ++k) {
        for (int j = 0; j <= 2; ++j) {
            for (int i = 0; i <= 2; ++i) {
                if (i == j) {
                    for (int n = 0; n <= 6; ++n) {
                        if (n == 1 || n == 4) {
                            temp = moments[7*(i*9+j*3+k)+n];
                            moments[7*(i*9+j*3+k)+n] = moments[7*(i*9+j*3+k)+n+1];
                            moments[7*(i*9+j*3+k)+n+1] = temp;
                        }
                    }
                } else if (i > j) {
                    for (int n = 0; n <= 6; ++n) {
                        if (n == 1 || n == 4) {
                            temp = moments[7*(i*9+j*3+k)+n];
                            moments[7*(i*9+j*3+k)+n] = moments[7*(j*9+i*3+k)+n+1];
                            moments[7*(j*9+i*3+k)+n+1] = temp;
                            temp = moments[7*(j*9+i*3+k)+n];
                            moments[7*(j*9+i*3+k)+n] = moments[7*(i*9+j*3+k)+n+1];
                            moments[7*(i*9+j*3+k)+n+1] = temp;
                        } else if (n == 0 || n == 3 || n == 6) {
                            temp = moments[7*(i*9+j*3+k)+n];
                            moments[7*(i*9+j*3+k)+n] = moments[7*(j*9+i*3+k)+n];
                            moments[7*(j*9+i*3+k)+n] = temp;
                        }
                    }
                }
            }
        }
    }
}

inline void reflect_moments_yz(double (&moments)[189]) {
    double temp;
    for (int k = 0; k <= 2; ++k) {
        for (int j = 0; j <= 2; ++j) {
            for (int i = 0; i <= 2; ++i) {
                if (j == k) {
                    for (int n = 0; n <= 6; ++n) {
                        if (n == 2 || n == 5) {
                            temp = moments[7*(i*9+j*3+k)+n];
                            moments[7*(i*9+j*3+k)+n] = moments[7*(i*9+j*3+k)+n+1];
                            moments[7*(i*9+j*3+k)+n+1] = temp;
                        }
                    }
                } else if (j > k) {
                    for (int n = 0; n <= 6; ++n) {
                        if (n == 2 || n == 5) {
                            temp = moments[7*(i*9+j*3+k)+n];
                            moments[7*(i*9+j*3+k)+n] = moments[7*(i*9+k*3+j)+n+1];
                            moments[7*(i*9+k*3+j)+n+1] = temp;
                            temp = moments[7*(i*9+k*3+j)+n];
                            moments[7*(i*9+k*3+j)+n] = moments[7*(i*9+j*3+k)+n+1];
                            moments[7*(i*9+j*3+k)+n+1] = temp;
                        } else if (n == 0 || n == 1 || n == 4) {
                            temp = moments[7*(i*9+j*3+k)+n];
                            moments[7*(i*9+j*3+k)+n] = moments[7*(i*9+k*3+j)+n];
                            moments[7*(i*9+k*3+j)+n] = temp;
                        }
                    }
                }
            }
        }
    }
}

inline void reflect_moments_xz(double (&moments)[189]) {
    double temp;
    for (int k = 0; k <= 2; ++k) {
        for (int j = 0; j <= 2; ++j) {
            for (int i = 0; i <= 2; ++i) {
                if (i == k) {
                    for (int n = 0; n <= 6; ++n) {
                        if (n == 1 || n == 4) {
                            temp = moments[7*(i*9+j*3+k)+n];
                            moments[7*(i*9+j*3+k)+n] = moments[7*(i*9+j*3+k)+n+2];
                            moments[7*(i*9+j*3+k)+n+2] = temp;
                        }
                    }
                } else if (i > k) {
                    for (int n = 0; n <= 6; ++n) {
                        if (n == 1 || n == 4) {
                            temp = moments[7*(i*9+j*3+k)+n];
                            moments[7*(i*9+j*3+k)+n] = moments[7*(k*9+j*3+i)+n+2];
                            moments[7*(k*9+j*3+i)+n+2] = temp;
                            temp = moments[7*(k*9+j*3+i)+n];
                            moments[7*(k*9+j*3+i)+n] = moments[7*(i*9+j*3+k)+n+2];
                            moments[7*(i*9+j*3+k)+n+2] = temp;
                        } else if (n == 0 || n == 2 || n == 5) {
                            temp = moments[7*(i*9+j*3+k)+n];
                            moments[7*(i*9+j*3+k)+n] = moments[7*(k*9+j*3+i)+n];
                            moments[7*(k*9+j*3+i)+n] = temp;
                        }
                    }
                }
            }
        }
    }
}
"""

print(reflect_subroutines, file=file)
print("} // namespace r2pnet\n", file=file)

file.close()