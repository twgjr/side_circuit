import ctypes

# DLL Function Prototypes
ngspice_dll = ctypes.cdll.LoadLibrary(r'.\ngspice_api\Spice64_dll\dll-vs\ngspice.dll')

# Define the struct types
class NgComplex(ctypes.Structure):
    _fields_ = [
        ("cx_real", ctypes.c_double),
        ("cx_imag", ctypes.c_double)
    ]

class VectorInfo(ctypes.Structure):
    _fields_ = [
        ("v_name", ctypes.c_char_p),
        ("v_type", ctypes.c_int),
        ("v_flags", ctypes.c_short),
        ("v_realdata", ctypes.POINTER(ctypes.c_double)),
        ("v_compdata", ctypes.POINTER(NgComplex)),
        ("v_length", ctypes.c_int)
    ]

class VecValues(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("creal", ctypes.c_double),
        ("cimag", ctypes.c_double),
        ("is_scale", ctypes.c_bool),
        ("is_complex", ctypes.c_bool)
    ]

class VecValuesAll(ctypes.Structure):
    _fields_ = [
        ("veccount", ctypes.c_int),
        ("vecindex", ctypes.c_int),
        ("vecsa", ctypes.POINTER(ctypes.POINTER(VecValues)))
    ]

class VecInfo(ctypes.Structure):
    _fields_ = [
        ("number", ctypes.c_int),
        ("vecname", ctypes.c_char_p),
        ("is_real", ctypes.c_bool),
        ("pdvec", ctypes.c_void_p),
        ("pdvecscale", ctypes.c_void_p)
    ]

class VecInfoAll(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("title", ctypes.c_char_p),
        ("date", ctypes.c_char_p),
        ("type", ctypes.c_char_p),
        ("veccount", ctypes.c_int),
        ("vecs", ctypes.POINTER(ctypes.POINTER(VecInfo)))
    ]

# Define callback function types
SendChar = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p)
SendStat = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p)
ControlledExit = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_bool, ctypes.c_int, ctypes.c_void_p)
SendData = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(VecValuesAll), ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
SendInitData = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(VecInfoAll), ctypes.c_int, ctypes.c_void_p)
BGThreadRunning = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_bool, ctypes.c_int, ctypes.c_void_p)

ngSpice_Init = ngspice_dll.ngSpice_Init
ngSpice_Init.argtypes = [SendChar, SendStat, ControlledExit, SendData, 
                         SendInitData, BGThreadRunning, ctypes.c_void_p]
ngSpice_Init.restype = ctypes.c_int

ngSpice_Command = ngspice_dll.ngSpice_Command
ngSpice_Command.argtypes = [ctypes.c_char_p]
ngSpice_Command.restype = ctypes.c_int

# Implement the callback functions
def send_char(string, lib_id, return_ptr):
    print(f"Received string from ngspice.dll (ID: {lib_id}): {string.decode()}")
    return 0

def send_stat(string, lib_id, return_ptr):
    print(f"Received status from ngspice.dll (ID: {lib_id}): {string.decode()}")
    return 0

def controlled_exit(exit_status, immediate_unload, exit_on_quit, lib_id, return_ptr):
    print(f"Received controlled exit signal from ngspice.dll (ID: {lib_id}): Exit Status: {exit_status}")
    return 0

def send_data(vec_values, count, lib_id, return_ptr):
    print(f"Received data values from ngspice.dll (ID: {lib_id})")
    for i in range(count):
        vec = vec_values.contents.vecsa[i]
        print(f"Vector Name: {vec.contents.name.decode()}, Real: {vec.contents.creal}, Imaginary: {vec.contents.cimag}")
    return 0

def send_init_data(vec_info, count, lib_id, return_ptr):
    print(f"Received initialization data from ngspice.dll (ID: {lib_id})")
    for i in range(count):
        vec = vec_info.contents.vecs[i]
        print(f"Vector Name: {vec.contents.vecname.decode()}, Is Real: {vec.contents.is_real}")
    return 0

def bg_thread_running(is_running, lib_id, return_ptr):
    print(f"Received background thread running status from ngspice.dll (ID: {lib_id}): Is Running: {is_running}")
    return 0

# Create callback function instances
send_char_callback = SendChar(send_char)
send_stat_callback = SendStat(send_stat)
controlled_exit_callback = ControlledExit(controlled_exit)
send_data_callback = SendData(send_data)
send_init_data_callback = SendInitData(send_init_data)
bg_thread_running_callback = BGThreadRunning(bg_thread_running)

# Call ngSpice_Init to initialize ngspice.dll
ngSpice_Init(send_char_callback, send_stat_callback, controlled_exit_callback,
             send_data_callback, send_init_data_callback, 
             bg_thread_running_callback, None)

# # Load circuit description file into ngspice
src_file = r".\working\rc_step.cir"
# src_file = r".\working\vdiv_step.cir"
source = "source"
source_cmd = source + " " + src_file
ngSpice_Command(source_cmd.encode('utf-8'))

# Run simulation
ngSpice_Command(b"run")

# Clean up and exit ngspice
ngSpice_Command(b"quit")