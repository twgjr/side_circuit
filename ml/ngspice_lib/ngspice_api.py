import ctypes
import os

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
SendChar = ctypes.CFUNCTYPE(ctypes.c_int, #return value
                            ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p)
SendStat = ctypes.CFUNCTYPE(ctypes.c_int, #return value
                            ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p)
ControlledExit = ctypes.CFUNCTYPE(ctypes.c_int, #return value
                    ctypes.c_int, ctypes.c_bool, ctypes.c_bool, ctypes.c_int, 
                    ctypes.c_void_p)
SendData = ctypes.CFUNCTYPE(ctypes.c_int, #return value
                ctypes.POINTER(VecValuesAll), ctypes.c_int, ctypes.c_int, 
                ctypes.c_void_p)
SendInitData = ctypes.CFUNCTYPE(ctypes.c_int, #return value
                    ctypes.POINTER(VecInfoAll), ctypes.c_int, ctypes.c_void_p)
BGThreadRunning = ctypes.CFUNCTYPE(ctypes.c_int, #return value
                        ctypes.c_bool, ctypes.c_int, ctypes.c_void_p)

class Spice():
    def __init__(self, dll_path:str) -> None:
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        # DLL Function Prototypes
        self.dll = ctypes.cdll.LoadLibrary(dll_path)

        self.ngspice_init = self.dll.ngSpice_Init
        self.ngspice_init.argtypes = [SendChar, SendStat, ControlledExit, SendData, 
                                SendInitData, BGThreadRunning, ctypes.c_void_p]
        self.ngspice_init.restype = ctypes.c_int

        self.ngspice_command = self.dll.ngSpice_Command
        self.ngspice_command.argtypes = [ctypes.c_char_p]
        self.ngspice_command.restype = ctypes.c_int

        # Create callback function instances
        self.send_char_callback = SendChar(self.send_char)
        self.send_stat_callback = SendStat(self.send_stat)
        self.controlled_exit_callback = ControlledExit(self.controlled_exit)
        self.send_data_callback = SendData(self.send_data)
        self.send_init_data_callback = SendInitData(self.send_init_data)
        self.bg_thread_running_callback = BGThreadRunning(self.bg_thread_running)

        # Call ngSpice_Init to initialize ngspice.dll
        self.ngspice_init(self.send_char_callback, self.send_stat_callback, 
                          self.controlled_exit_callback,self.send_data_callback,
                          self.send_init_data_callback, 
                          self.bg_thread_running_callback, None)
        
    # Implement the callback functions
    @staticmethod
    def send_char(string, lib_id, return_ptr):
        print(f"Received string from ngspice.dll (ID: {lib_id}): {string.decode()}")
        return 0

    @staticmethod
    def send_stat(string, lib_id, return_ptr):
        print(f"Received status from ngspice.dll (ID: {lib_id}): {string.decode()}")
        return 0

    @staticmethod
    def controlled_exit(exit_status, immediate_unload, exit_on_quit, lib_id, return_ptr):
        print(f"Received controlled exit signal from ngspice.dll (ID: {lib_id}): Exit Status: {exit_status}")
        return 0

    @staticmethod
    def send_data(vec_values_all, count, lib_id, return_ptr):
        print(f"Received data values from ngspice.dll (ID: {lib_id})")
        for i in range(count):
            vec_value = vec_values_all.contents.vecsa[i]
            name = vec_value.contents.name.decode()
            real = vec_value.contents.creal
            imag = vec_value.contents.cimag
            print(f"Vector Name: {name}, Real: {real}, Imaginary: {imag}")
        return 0

    @staticmethod
    def send_init_data(vec_info_all, lib_id, return_ptr):
        print(f"Received initialization data from ngspice.dll (ID: {lib_id})")
        for i in range(vec_info_all.contents.veccount):
            vec_info = vec_info_all.contents.vecs[i]
            name = vec_info.contents.vecname.decode()
            is_real = vec_info.contents.is_real
            print(f"Vector Name: {name}, Is Real: {is_real}")
        return 0

    @staticmethod
    def bg_thread_running(is_running, lib_id, return_ptr):
        print(f"Received background thread running status from ngspice.dll (ID: {lib_id}): Is Running: {is_running}")
        return 0

    def source(self, cir_path:str):
        src = "source"
        source_cmd = src + " " + cir_path
        self.ngspice_command(source_cmd.encode('utf-8'))

    def circ_by_line(self, circ_line:str):
        '''BASIC RC CIRCUIT AC ANALYSIS
            circbyline r 1 2 1.0
            circbyline c 2 0 1.0
            circbyline vin 1 0 dc 0 ac 1 $ <--- the ac source
            circbyline .options noacct
            circbyline .ac dec 10 .01 10
            circbyline .end
        '''
        cbl_cmd = "circbyline"
        command = cbl_cmd + " " + circ_line
        self.ngspice_command(command.encode('utf-8'))

    def trans(self, tstep:float, tstop:float, tmax:float=None, use_ic=False):
        line = ".tran "+str(tstep)+" "+str(tstop)+" "+str(0)
        if(tmax != None):
            line += " "+str(tmax)
        if(use_ic == True):
            line += " uic"
        self.circ_by_line(line)

    def end_circuit(self):
        self.circ_by_line(".end")

    def run(self):
        self.ngspice_command(b"run")

    def quit(self):
        self.ngspice_command(b"quit")

if(__name__=='__main__'):
    spice = Spice(dll_path=r'.\ngspice_lib\Spice64_dll\dll-vs\ngspice.dll')
    spice.source(cir_path=r".\working\rc_step.cir")
    spice.run()
    spice.quit()