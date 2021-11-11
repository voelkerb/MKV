
import sys
import os
import numpy as np
import av
import math
from typing import Union, List, Tuple, Optional
from pysubs2 import SSAFile, SSAEvent

import time

class bcolors:
    """Enumeration class for escape characters in different colors"""
    HEADER = '\033[95m'
    OK = '\033[94m'
    OK2 = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

VERBOSE = False
# TODO: All LOAD-Functions crash with Seg Fault when containing more than 7 Streams
# Has been nailed down to "to_ndarray" array function or "av.io.read"

def __to_milliseconds(timestring):
    """
    Convert a given timestring (basically a duration) to milliseconds.

    Required timestring format is "<hours>:<min>:<seconds>.<ms>" e.g. "00:04:20.0"

    :param timestring:     timestring (duration), format: "00:04:20.0"
    :type  timestring:     str

    :return: Duration as number of milliseconds
    :rtype:  int
    """
    t = timestring.split(".")
    if len(t[1]) == 1:   ms = int(t[1])*100
    elif len(t[1]) == 2: ms = int(t[1])*10
    elif len(t[1]) == 3: ms = int(t[1])
    t = t[0].split(":")
    h = int(t[0])
    m = int(t[1])
    s = int(t[2])
    return ms + s*1000 + m*1000*60 + h*1000*60*60
 

def streamsForTitle(streams, titles):
    """Extract only streams that match the given title"""
    # Handle title not list case
    if not isinstance(titles, list): titles = [titles]
    # Loop over stream metadata and look for title match
    newStreams = []
    for stream in streams:
        key =  set(["Title", "title", "TITLE", "NAME", "Name", "name"]).intersection(set(stream.metadata.keys()))
        if len(key) > 0:
            title = stream.metadata[next(iter(key))]
            if title in titles: newStreams.append(stream)
    return newStreams

# Only supports audio
def seekLoad(filepath, seekPos, length, verbose=False, streamsToLoad=None, titles=None):
    r"""
    Call to load part of an MKV file.

    As filepath, give the mkv file to load into an array of dictionaries.
    Each dictionary holds information such as metadata, title, name
    dataList has the following structure:

    .. code-block:: python3

        dataList = [
            {
                title:        < streamTitle >,
                startTime:    < start of the data passed in ms >,
                endTime:      < end of the data passed in ms >,
                streamIndex:  < index of the stream >,
                metadata:     < metadata of the stream >,
                type:         < type of the stream >,
                samplingrate: < samplingrate >,                       # only for audio
                measures:     < list of measures >,                   # only for audio
                data:         < data as recarray with fieldnames >,   # only for audio
            },
            ...
        ]

    :param filepath:       filepath to the mkv to open
    :type  filepath:       str
    :param seekPos:        Start position in seconds
    :type  seekPos:        float
    :param length:         Length in secodns to load from start position
    :type  length:         float
    :param verbose:        increase output verbosity
    :type  verbose:        Bool
    :param streamsToLoad:  List of streams to load from the file, either list of numbers or list of stream titles
                           default: None -> no fileter based on streamsToLoad
    :type  streamsToLoad:  list
    :param titles:         List or str of streams titles which should be loaded
                           default: None -> no filter based on title names
    :type  titles:         list or str

    :return: List of dictionaries holding data and info
    :rtype: list
    """
    import pysubs2
    if seekPos < 0:
        print(bcolors.WARNING + "seekpos smaller 0" + str(seekPos) + bcolors.ENDC)
        seekPos = 0
    # PTS is in milliseconds
    start_pts = int(seekPos*1000)
    end_pts = start_pts + int(length*1000)

    if verbose: print(bcolors.WARNING + "Seek from: " + str(start_pts) + "ms to " + str(end_pts) + "ms" + bcolors.ENDC)

    # Open the container
    try: container = av.open(filepath)
    except av.AVError: return []
    # Get available stream indices from file
    streams = [s for i, s in enumerate(container.streams) if (s.type == 'audio' or s.type == 'subtitle' or s.type == 'video')]
    # Look if it is a stream to be loaded
    if isinstance(streamsToLoad, int): streamsToLoad = [streamsToLoad]
    if streamsToLoad is not None: streams = [stream for stream in streams if stream.index in streamsToLoad]
    if titles is not None: streams = streamsForTitle(streams, titles)
    if len(streams) == 0: return []
    # Copy over stream infos into datalist
    dataDict = {i["streamIndex"]:i for i in info(filepath)["streams"] if i["streamIndex"] in [s.index for s in streams]}
    # Verbose print
    if verbose:
        for key, stream in dataDict.items():
            print(bcolors.OK + "Stream : " + str(stream["streamIndex"]) + bcolors.ENDC)
            for k in ["type", "title", "metadata", "samplingrate", "measures", "duration"]:
                if k in stream: print("{}: {}".format(k, stream[k]))
    # Prepare dict for data
    for i in dataDict.keys():
        if dataDict[i]["type"] == "subtitle":
            dataDict[i]["subs"] = pysubs2.SSAFile()
        # Audio data will be stored here
        startSample = int(start_pts/1000.0*dataDict[i]["samplingrate"])
        stopSample = min(int(end_pts/1000.0*dataDict[i]["samplingrate"]), dataDict[i]["samples"])
        dataDict[i]["data"] = np.empty((stopSample-startSample,len(dataDict[i]["measures"])), dtype=np.float32)
        dataDict[i]["storeIndex"] = 0 
    # Need to load container again here
    try: container = av.open(filepath)
    except av.AVError: return []

    for stream in streams:
        index = stream.index
        if verbose: print(stream)
        container = av.open(filepath)
        container.seek(start_pts, whence='time', any_frame=False, stream=stream)
        initCheck = False
        for frame in container.decode(streams=stream.index):
            # print(frame.sample_rate)
            # print(frame.samples)
            # Look if we can seek to next frame
            if frame.pts + int(frame.samples/float(frame.sample_rate)*1000) < start_pts: continue
            if frame.pts > end_pts: break
            if not initCheck:
                if frame.pts > start_pts:
                    print(bcolors.FAIL + "seeked too far " + str(frame.pts) + " instead of " + str(start_pts)+ bcolors.ENDC)
                    sys.exit()
                initCheck = True
            if verbose: print(frame.pts)
            # If we need to skip data at the beginning, 0 else
            sliceStart = max(0, int((start_pts-frame.pts)/1000.0*frame.sample_rate))
            # If we need to skip data at the end
            sliceEnd = min(frame.samples, int((end_pts-frame.pts)/1000.0*frame.sample_rate))
            if verbose: print("slice: " + str(sliceStart) + " -> " + str(sliceEnd))
            ndarray = frame.to_ndarray().transpose()[sliceStart:sliceEnd]
            j = dataDict[index]["storeIndex"]
            dataDict[index]["data"][j:j+ndarray.shape[0],:] = ndarray[:,:]
            dataDict[index]["storeIndex"] += ndarray.shape[0]

    # Make recarray from data
    for i in dataDict.keys():
        if dataDict[i]["type"] == 'audio' and len(dataDict[i]["data"]) > 0:
            dataDict[i]["data"] = np.core.records.fromarrays(dataDict[i]["data"][:dataDict[index]["storeIndex"]].transpose(), dtype={'names': dataDict[i]["measures"], 'formats': ['f4']*len(dataDict[i]["measures"])})
            del dataDict[index]["storeIndex"]
    # RETURN it as a list
    return [dataDict[i] for i in dataDict.keys()]

def loadAudio(filepath: str, streamsToLoad: Optional[List[int]]=None, titles: Optional[List[str]]=None, start: Optional[float]=0, duration: Optional[float]=-1) -> List[dict]:
    r"""
    Call to load audio from an MKV file to an numpy array the fast way.
    Note: All streams MUST have the same samplingrate!

    As filepath, give the mkv file to load into an array of dictionaries.
    Each dictionary holds information such as metadata, title, name
    dataList has the following structure:

    .. code-block:: python3

        dataList = [
            {
                title:        < streamTitle >,
                streamIndex:  < index of the stream >,
                metadata:     < metadata of the stream >,
                type:         < type of the stream >,
                samplingrate: < samplingrate >,                       # only for audio
                measures:     < list of measures >,                   # only for audio
                data:         < data as recarray with fieldnames >,   # only for audio
            },
            ...
        ]

    :param filepath:       filepath to the mkv to open
    :type  filepath:       str
    :param start:          start time in the file (seconds from file start)
                           default: 0
    :type  start:          float
    :param duration:       Duration to load data (from start)
                           default: -1 (all data)
    :type  duration:       float
    :param streamsToLoad:  List of streams to load from the file, either list of numbers or list of stream titles
                           default: None -> all streams should be loaded
    :type  streamsToLoad:  list
    :param titles:         List or str of streams titles which should be loaded
                           default: None -> no filter based on title names
    :type  titles:         list or str

    :return: List of dictionaries holding data and info
    :rtype: list
    """
    # Open the container
    try: container = av.open(filepath)
    except av.AVError: return []
    # Get available stream indices from file
    streams = [s for i, s in enumerate(container.streams) if ( (s.type == 'audio') )]
    # Look if it is a stream to be loaded
    if isinstance(streamsToLoad, int): streamsToLoad = [streamsToLoad]
    if streamsToLoad is not None: streams = [stream for stream in streams if stream.index in streamsToLoad]
    if titles is not None: streams = streamsForTitle(streams, titles)
    if len(streams) == 0: return []
    # Copy over stream infos into datalist
    dataDict = {i["streamIndex"]:i for i in info(filepath)["streams"] if i["streamIndex"] in [s.index for s in streams]}
    # VERBOSE print
    if VERBOSE:
        for key, stream in dataDict.items():
            print("Stream : " + str(stream["streamIndex"]))
            for k in ["type", "title", "metadata", "samplingrate", "measures", "duration"]:
                if k in stream: print("{}: {}".format(k, stream[k]))
    indices = [stream.index for stream in streams]
    # Prepare dict for data
    for i in dataDict.keys():
        # Allocate empty array
        dur = dataDict[i]["samples"]/dataDict[i]["samplingrate"]-start
        if duration >= 0: dur = min(dur, duration)
        samples = max(0, int(math.ceil(dur*float(dataDict[i]["samplingrate"]))))
        dataDict[i]["samples"] = samples
        dataDict[i]["duration"] = dur
        dataDict[i]["timestamp"] = dataDict[i]["timestamp"]+start
        dataDict[i]["data"] = np.ones((dataDict[i]["samples"],len(dataDict[i]["measures"])), dtype=np.float32)
        # dataDict[i]["data"] = np.empty((dataDict[i]["samples"],len(dataDict[i]["measures"])), dtype=np.float32)
        dataDict[i]["storeIndex"] = 0 
    
    start_pts = start*1000.0
    end_pts = start_pts + duration*1000
    inited = False

    for stream in streams:
        index = stream.index
        if VERBOSE: print(stream)
        container = av.open(filepath)
        container.seek(math.floor(start_pts), whence='time', any_frame=False, stream=stream)
        initCheck = False
        for frame in container.decode(streams=stream.index):
            # Check seek status
            if not inited:
                if frame.pts > start_pts: 
                    raise AssertionError("Seeked too far, should not happen: {}ms - {}ms".format(frame.pts, start_pts))
                    pass
            # Check start 
            if frame.pts + int(frame.samples/float(frame.sample_rate)*1000) < start_pts: continue
            # Check end
            if duration != -1 and frame.pts > end_pts: break
        
            # If we need to skip data at the beginning, 0 else
            # This is only ms resolution
            # NOTE: Does this cause problems for us?? we need to find out
            s = max(0, math.floor(float(start_pts-float(frame.pts))/1000.0*frame.sample_rate))
            # so use samplecount
            #s = max(0, int(frame.samples - (dataDict[index]["samples"] - dataDict[index]["storeIndex"])))
            # If we need to skip data at the end
            if duration >= 0: e = min(frame.samples, float((end_pts-frame.pts)/1000.0*frame.sample_rate))
            else: e = frame.samples
            # If there were rounding issues
            e = min(e, int(s+dataDict[index]["samples"] - dataDict[index]["storeIndex"]))

            # if not inited:
            #     print("dur: {}, endPts: {}".format(duration, end_pts, ))
            #     print("{}:{} - f:{}, sr: {}, pts:{}, start:{}, {}".format(s,e, frame.samples,frame.sample_rate, frame.pts, start_pts, (start_pts-frame.pts)/1000.0*frame.sample_rate))
            # Get corresponding index in dataList array
            ndarray = frame.to_ndarray().transpose()[int(s):int(e),:]
            # copy over data
            j = dataDict[index]["storeIndex"]
            dataDict[index]["data"][j:j+ndarray.shape[0],:] = ndarray[:,:]
            dataDict[index]["storeIndex"] += ndarray.shape[0]
            if not inited:
                inited = True
   
    # Demultiplex the individual packets of the file
    # for i, packet in enumerate(container.demux(streams)):
        
    #     # Inside the packets, decode the frames
    #     for frame in packet.decode(): # codec_ctx.decode(packet):

    #         # TODO:
    #         if not inited:
    #             inited = True
    #             if frame.pts > start_pts: raise AssertionError("Seeked too far, should not happen: {}ms - {}ms".format(frame.pts, start_pts))
    #             # Look if we can seek to next frame
    #         if frame.pts + int(frame.samples/float(frame.sample_rate)*1000) < start*1000: continue
    #         if stop != -1 and frame.pts > stop*1000: break
    #         # If we need to skip data at the beginning, 0 else
    #         s = max(0, int((start*1000-frame.pts)/1000.0*frame.sample_rate))
    #         # If we need to skip data at the end
    #         if stop != -1: e = min(frame.samples, int((stop*1000-frame.pts)/1000.0*frame.sample_rate))
    #         else: e = frame.samples

    #         # Get corresponding index in dataList array
    #         index = packet.stream.index
    #         ndarray = frame.to_ndarray().transpose()
    #         # copy over data
    #         j = dataDict[index]["storeIndex"]
    #         dataDict[index]["data"][j:j+ndarray.shape[0],:] = ndarray[:,:]
    #         dataDict[index]["storeIndex"] += ndarray.shape[0]
    # Make recarray from data
    for i in dataDict.keys():
        if "storeIndex" in dataDict[i]: del dataDict[i]["storeIndex"] 
        dataDict[i]["data"] = np.core.records.fromarrays(dataDict[i]["data"].transpose(), dtype={'names': dataDict[i]["measures"], 'formats': ['f4']*len(dataDict[i]["measures"])})
    # RETURN it as a list
    return [dataDict[i] for i in dataDict.keys()]


def chunkLoads(fileList, timeslice, starttime=0, stoptime=-1, verbose=False, streamsToLoad=None, titles=None):
    r"""
    Load multiple files in chunks to keep memory usage small.
    The purpose is to have a dataset split into multiple files.
    fileList must be sorted!!!

    Yields a list of dictionaries.
    Each dictionary holds information such as metadata, title, name and data.
    The list has the following structure:

    .. code-block:: python3

        dataList = [
            {
                title:        < streamTitle >,
                streamIndex:  < index of the stream >,
                metadata:     < metadata of the stream >,
                type:         < type of the stream >,
                samplingrate: < samplingrate >,                       # only for audio
                measures:     < list of measures >,                   # only for audio
                data:         < data as recarray with fieldnames >,   # only for audio
            },
            ...
        ]

    :param fileList:       List of filepath to the mkvs to open. Must be sorted and continues
    :type  fileList:       list of str
    :param timeslice:      timeslice that is returned
    :type  timeslice:      float
    :param verbose:        increase output verbosity
    :type  verbose:        Bool
    :param streamsToLoad:  List of streams to load from the file, either list of numbers or list of stream titles
                           default: None -> all streams should be loaded
    :type  streamsToLoad:  list
    :param titles:         List or str of streams titles which should be loaded
                           default: None -> no filter based on title names
    :type  titles:         list or str

    :yield: List of dictionaries holding data and info
    :rtype: list
    """
    finished = False
    start = starttime
    duration = stoptime-starttime
    currentDuration = 0
    missingSeconds = 0
    for j, file in enumerate(fileList):
        # Skipping files out of starttime
        inf = info(file)["streams"][0]
        if inf["duration"] < start:
            start -= inf["duration"]
            continue
        for dataListChunk in chunkLoad(file, timeslice, starttime=start, verbose=verbose, streamsToLoad=streamsToLoad, titles=titles):
            # We started from starttime frist, now we have to set
            start = missingSeconds
            # Check if chunk length matches
            addNext = any([int(timeslice*s["samplingrate"]) > s["samples"] for s in dataListChunk])
            # If not, load chunk from next file if there is one left
            if addNext and file != fileList[-1]:
                # Calculate missing seconds
                missingSeconds = timeslice - (dataListChunk[0]["samples"]/dataListChunk[0]["samplingrate"])
                start = missingSeconds
                # load one chunk from nextfile
                addedChunk = next(chunkLoad(fileList[j+1], missingSeconds, starttime=0, verbose=verbose, streamsToLoad=streamsToLoad, titles=titles))
                # Copy data over and set samples and duration
                for i in range(len(dataListChunk)):
                    dataListChunk[i]["data"] = np.concatenate((dataListChunk[i]["data"], addedChunk[i]["data"]))
                    dataListChunk[i]["samples"] += addedChunk[i]["samples"]
                    dataListChunk[i]["duration"] += addedChunk[i]["duration"]
            # If we have a stop time, clean data at the end, if chunk was too much
            if stoptime != -1:
                dis = (currentDuration + dataListChunk[0]["duration"]) - duration
                # Look if stop has been reached
                if dis >= 0:
                    # Indicate finish
                    finished = True
                    # Clean data
                    for i in range(len(dataListChunk)):
                        endSample = len(dataListChunk[i]["data"])-int(dis*dataListChunk[i]["samplingrate"])
                        dataListChunk[i]["data"] = dataListChunk[i]["data"][:endSample]
                        dataListChunk[i]["samples"] = len(dataListChunk[i]["data"])
                        dataListChunk[i]["duration"] = dataListChunk[i]["samples"]/dataListChunk[i]["samplingrate"]

            currentDuration += dataListChunk[0]["duration"]
            yield dataListChunk
            if finished: return




def chunkLoad(filepath, timeslice, starttime=0, stoptime=-1, verbose=False, streamsToLoad=None, titles=None):
    r"""
    Load a file in chunks to keep memory usage small.

    Yields a list of dictionaries.
    Each dictionary holds information such as metadata, title, name and data.
    The list has the following structure:

    .. code-block:: python3

        dataList = [
            {
                title:        < streamTitle >,
                streamIndex:  < index of the stream >,
                metadata:     < metadata of the stream >,
                type:         < type of the stream >,
                samplingrate: < samplingrate >,                       # only for audio
                measures:     < list of measures >,                   # only for audio
                data:         < data as recarray with fieldnames >,   # only for audio
            },
            ...
        ]

    :param filepath:       filepath to the mkv to open
    :type  filepath:       str
    :param timeslice:      timeslice that is returned
    :type  timeslice:      float
    :param verbose:        increase output verbosity
    :type  verbose:        Bool
    :param streamsToLoad:  List of streams to load from the file, either list of numbers or list of stream titles
                           default: None -> all streams should be loaded
    :type  streamsToLoad:  list
    :param titles:         List or str of streams titles which should be loaded
                           default: None -> no filter based on title names
    :type  titles:         list or str

    :yield: List of dictionaries holding data and info
    :rtype: list
    """
    chunkSizes, dataOnly, loadedFrames, dataLen, dataDict = {}, {}, {}, {}, {}
    # Open the container
    try: container = av.open(filepath)
    except av.AVError: return []
    # NOTE: As for now, this only works for audio data
    # Get available stream indices from file
    streams = [s for i, s in enumerate(container.streams) if ( (s.type == 'audio') )]
    # Get available stream indices from file
    if isinstance(streamsToLoad, int): streamsToLoad = [streamsToLoad]
    if streamsToLoad is not None: streams = [stream for stream in streams if stream.index in streamsToLoad]
    if titles is not None: streams = streamsForTitle(streams, titles)
    if len(streams) == 0: return []
    # Copy over stream infos into datalist
    dataDict = {i["streamIndex"]:i for i in info(filepath)["streams"] if i["streamIndex"] in [s.index for s in streams]}
    
    # Prepare chunk size dict
    for i in dataDict.keys(): 
        chunkSizes[i] = int(dataDict[i]["samplingrate"]*timeslice)
        # Audio data will be stored here
        dataOnly[i] = np.empty((chunkSizes[i],len(dataDict[i]["measures"])), dtype=np.float32)
        loadedFrames[i] = []
        dataLen[i] = 0
    
    chunks = 0
    # Copy over timestamps so the timestamp of each chunk can be calculated
    timestamps = {i:dataDict[i]["timestamp"] for i in dataDict}
    # De-multiplex the individual packets of the file
    for packet in container.demux(streams):
        i = packet.stream.index
            
        # Inside the packets, decode the frames
        for frame in packet.decode():

            # Look if we can seek to next frame
            if frame.pts + int(frame.samples/float(frame.sample_rate)*1000) < starttime*1000: continue
            if stoptime != -1 and frame.pts > stoptime*1000: break
            # If we need to skip data at the beginning, 0 else
            s = max(0, int((starttime*1000-frame.pts)/1000.0*frame.sample_rate))
            # If we need to skip data at the end
            if stoptime != -1: e = min(frame.samples, int((stoptime*1000-frame.pts)/1000.0*frame.sample_rate))
            else: e = frame.samples

            # The frame can be audio, subtitles or video
            if packet.stream.type == 'audio':
                ndarray = frame.to_ndarray().T[s:e]
                loadedFrames[i].append(ndarray)
                dataLen[i] += ndarray.shape[0]
                
        # If the chunks have been loaded for all streams (sometimes chunksize smaller the framesize -> hence while)
        while all([dataLen[index] >= chunkSizes[index] for index in chunkSizes.keys()]):
            dataReturn = []
            for i in chunkSizes.keys():
                # Copy over metadata of stream
                dataReturn.append(dataDict[i])

                # Copy data of frames into chunks
                currentLen = 0
                while currentLen < chunkSizes[i]:
                    end = min(loadedFrames[i][0].shape[0], chunkSizes[i]-currentLen)
                    dataOnly[i][currentLen:currentLen+end] = loadedFrames[0][i][:end]
                    currentLen += end
                    loadedFrames[i][0] = loadedFrames[i][0][end:]
                    if loadedFrames[i][0].shape[0] == 0: del loadedFrames[i][0]
                
                dataReturn[-1]["samples"] = dataOnly[i][:currentLen].shape[0]
                dataReturn[-1]["duration"] = dataReturn[-1]["samples"]/dataReturn[-1]["samplingrate"]
                dataReturn[-1]["timestamp"] = timestamps[i] + starttime + chunks*chunkSizes[i]/dataReturn[-1]["samplingrate"]
                dataReturn[-1]["data"] = np.core.records.fromarrays(dataOnly[i][:currentLen].T, dtype={'names': dataReturn[-1]["measures"], 'formats': ['f4']*len(dataReturn[-1]["measures"])})
                dataLen[i] = max(0, dataLen[i]-chunkSizes[i])
            chunks += 1
            yield dataReturn

    # This is for the remaining chunk that is smaller than chunksize
    while not all([dataLen[index] == 0 for index in chunkSizes.keys()]):
        dataReturn = []
        for i in chunkSizes.keys():
            # Copy over metadata of stream
            dataReturn.append(dataDict[i])
            # Copy data of frames into chunks
            currentLen = 0
            while currentLen < min(chunkSizes[i], dataLen[i]):
                end = min(loadedFrames[i][0].shape[0], chunkSizes[i]-currentLen)
                dataOnly[i][currentLen:currentLen+end] = loadedFrames[0][i][:end]
                currentLen += end
                loadedFrames[i][0] = loadedFrames[i][0][end:]
                if loadedFrames[i][0].shape[0] == 0: del loadedFrames[i][0]
            dataReturn[-1]["samples"] = dataOnly[i][:currentLen].shape[0]
            dataReturn[-1]["duration"] = dataReturn[-1]["samples"]/dataReturn[-1]["samplingrate"]
            dataReturn[-1]["timestamp"] = timestamps[i] + starttime + chunks*chunkSizes[i]/dataReturn[-1]["samplingrate"]
            dataReturn[-1]["data"] = np.core.records.fromarrays(dataOnly[i][:currentLen].T, dtype={'names': dataReturn[-1]["measures"], 'formats': ['f4']*len(dataReturn[-1]["measures"])})
            dataLen[i] = max(0, dataLen[i]-chunkSizes[i])
        yield dataReturn

def load(filepath, verbose=False, streamsToLoad=None, audio=True, video=False, subs=True, titles=None):#, measuresToLoad=None):
    r"""
    Call to load an MKV file.

    As filepath, give the mkv file to load into an array of dictionaries.
    Each dictionary holds information such as metadata, title, name
    dataList has the following structure:

    .. code-block:: python3

        dataList = [
            {
                title:        < streamTitle >,
                streamIndex:  < index of the stream >,
                metadata:     < metadata of the stream >,
                type:         < type of the stream >,
                samplingrate: < samplingrate >,                       # only for audio
                measures:     < list of measures >,                   # only for audio
                data:         < data as recarray with fieldnames >,   # only for audio
                subs:         < subtitles as pysubs2 file >,          # only for subtitles
            },
            ...
        ]

    :param filepath:       filepath to the mkv to open
    :type  filepath:       str
    :param verbose:        increase output verbosity
    :type  verbose:        Bool
    :param audio:          Load audio files
    :type  audio:          Bool
    :param video:          Load video files
    :type  video:          Bool
    :param subs:           Load subtitle files
    :type  subs:           Bool
    :param streamsToLoad:  List of streams to load from the file, either list of numbers or list of stream titles
                           default: None -> all streams should be loaded
    :type  streamsToLoad:  list
    :param measuresToLoad: List of measures to load from the file, according to valuesAndUnits file
                           default: None -> all measures should be loaded
    :type  measuresToLoad: list

    :return: List of dictionaries holding data and info
    :rtype: list
    """
    import pysubs2
    # Open the container
    try: container = av.open(filepath)
    except av.AVError: return []
    # Get available stream indices from file
    streams = [s for i, s in enumerate(container.streams) if
        ( (s.type == 'audio' and audio) or (s.type == 'subtitle' and subs) or (s.type == 'video' and video) )
    ]
    # Look if it is a stream to be loaded
    if isinstance(streamsToLoad, int): streamsToLoad = [streamsToLoad]
    if streamsToLoad is not None: streams = [stream for stream in streams if stream.index in streamsToLoad]
    if titles is not None: streams = streamsForTitle(streams, titles)
    if len(streams) == 0: return []
    # Copy over stream infos into datalist
    dataDict = {i["streamIndex"]:i for i in info(filepath)["streams"] if i["streamIndex"] in [s.index for s in streams]}
    # Verbose print
    if verbose:
        for _, stream in dataDict.items():
            print(bcolors.OK + "Stream : " + str(stream["streamIndex"]) + bcolors.ENDC)
            for k in ["type", "title", "metadata", "samplingrate", "measures", "duration"]:
                if k in stream: print("{}: {}".format(k, stream[k]))
    # Prepare dict for data
    for i in dataDict.keys():
        if dataDict[i]["type"] == "subtitle":
            dataDict[i]["subs"] = pysubs2.SSAFile()
        if dataDict[i]["type"] == "audio":
            # Allocate empty array
            dataDict[i]["data"] = np.empty((dataDict[i]["samples"],len(dataDict[i]["measures"])), dtype=np.float32)
            dataDict[i]["storeIndex"] = 0 
    # Demultiplex the individual packets of the file
    for i, packet in enumerate(container.demux(streams)):
        # Inside the packets, decode the frames
        for frame in packet.decode(): # codec_ctx.decode(packet):
            # Get corresponding index in dataList array
            index = packet.stream.index
            # The frame can be audio, subtitles or video
            if packet.stream.type == 'video': pass                 
            elif packet.stream.type == 'audio':
                ndarray = frame.to_ndarray().transpose()
                # copy over data
                j = dataDict[index]["storeIndex"]
                dataDict[index]["data"][j:j+ndarray.shape[0],:] = ndarray[:,:]
                dataDict[index]["storeIndex"] += ndarray.shape[0]
            if packet.stream.type == 'subtitle':
                for rect in frame.rects:
                    if rect.type == b'srt':
                        raise "To implement"
                    if rect.type == b'ass':
                        subList = rect.ass.lstrip("Dialogue: ").rstrip("\r\n").split(",")
                        if len(subList) == 10:
                            start = __to_milliseconds(subList[1])
                            stop = __to_milliseconds(subList[2])
                            dataDict[index]["subs"].append(pysubs2.SSAEvent(start=pysubs2.make_time(s=start/1000.0), end=pysubs2.make_time(s=stop/1000.0), text=subList[-1]))
    # Make recarray from data
    for i in dataDict.keys():
        if "storeIndex" in dataDict[i]:
            del dataDict[i]["storeIndex"] 
        if dataDict[i]["type"] == 'audio':
            dataDict[i]["data"] = np.core.records.fromarrays(dataDict[i]["data"].transpose(), dtype={'names': dataDict[i]["measures"], 'formats': ['f4']*len(dataDict[i]["measures"])})
        if dataDict[i]["type"] == 'subtitle':
            if verbose: print(dataDict[i]["subs"].to_string("srt"))
    # RETURN it as a list
    return [dataDict[i] for i in dataDict.keys()]

def mapSubsToAudio(dataList, fallBackMapping=False):
    """
    Map Subtitles to audio list.

    If you have loaded an MKV with subtitels and audio files,
    this file will map the subtitle file based on its name to the audio streams.
    If fallbackMapping is selected, the remaining files, will be mapped ascending to the last
    audio streams: (a0:; a1:s0; a2:s1; a3:s2; for 3 subtitles and 4 audio files). This is handy,
    if the first stream is the aggregated stream and all others have subtitles but names do not match.

    :param dataList:        must be loaded with mkv.load()
    :type  dataList:        list of dictionaries
    :param fallBackMapping: Use Fallback mapping to map the subs ascending to the last audio files. default=False
    :type  fallBackMapping: bool

    :return: tuple(0) List of dictionaries holding data and mapped subtitles according to mkv.load
             tuple(1) List of remaining subtitles which could not be mapped. Could be empty
    :rtype:  tuple of size 2 (both lists)
    """
    subsList = [dataDict for dataDict in dataList if dataDict["type"] == "subtitle"]
    subsFoundList = [False for i in range(len(subsList))]
    audioList = [dataDict for dataDict in dataList if dataDict["type"] == "audio"]
    for i, audioDict in enumerate(audioList):
        for j, subsDict in enumerate(subsList):
            if audioDict["title"] == subsDict["title"]:
                audioList[i]["subs"] = subsDict["subs"]
                subsFoundList[j] = True
                break
    if fallBackMapping:
        remains = [subsList[i] for i,b in enumerate(subsFoundList) if b == False]
        index = len(remains) - 1
        for i, audio in reversed(list(enumerate(audioList))):
            if "subs" not in audio and index >= 0:
                audioList[i]["subs"] = remains[index]["subs"]
                index -= 1

    return audioList, [subsList[i] for i,b in enumerate(subsFoundList) if b == False]


def __call_ffmpeg(call, verbose=False):
    """
    Call ffmpeg process in subprocess.

    Output from ffmpeg call is displayed using prefix $ffmpeg:

    :param call:       ffmpeg call to execute.
    :type  call:       str
    """
    import subprocess
    #  Construct all ffmpeg calls

    if verbose:
        OUT = subprocess.PIPE
        print("ffmpegCall: " + str(call))
        # process = subprocess.Popen(call, shell=True, stdin=subprocess.PIPE, 
        #                            preexec_fn=os.setsid)
    else:

        # This removes errors such as k XXX is too large
        split = call.split(" ")
        split.insert(1, "-loglevel panic") 
        call = " ".join(split)
        OUT = open(os.devnull, 'w')

    process = subprocess.Popen(call, shell=True, stdin=subprocess.PIPE, stdout=OUT, stderr=OUT,
                            preexec_fn=os.setsid, universal_newlines=True)
    # Read the output of the process line by line and output it to the console
    if verbose:
        import select
        while process.poll() == None:
            readable, _, _ = select.select([process.stderr, process.stdout], [], [])
            for reader in readable:
                line = reader.readline()
                if line:
                    print("$ffmeg: " + str(line.rstrip()))
    process.wait()


def makeMetaArgsFromDict(dic, type, stream=0):
    """
    Return given metadata as ffmpeg call arguments.

    Constructed metadata arguments from given title type and measures and stream.
    This arguments can be passed to ffmpegf call (e.g. as extraArgs - see other functions in this class).

    :param title:       Title added to the stream
    :type  title:       str
    :param type:        Type of the stream (audio, subtitle/subs, video etc.)
    :type  type:        str
    :param measures:    samplingrate of the given data
    :type  measures:    list
    :param stream:      potential extra arguments for ffmpeg. Consider to add ChannelNames here, default="".
    :type  stream:      int

    :return: metadata for the stream as ffmpeg arguments
    :rtype:  str
    """
    if type[0] not in ["s", "a", "v"]:
        print(bcolors.FAIL + "Currently only \"audio\", \"subtitles\" or \"video\" is allowed as type"+ bcolors.ENDC)
        return ""
    meta = " -metadata:s:" + str(type[0]) + ":" + str(stream)
    call = ""

    if "measures" in dic: 
        call += meta + " CHANNELS=" + str(len(dic["measures"])) + meta + " CHANNEL_TAGS=\""
        call += ",".join(dic["measures"]) + "\""
    if "title" in dic: 
        call += meta + " " + "title" + "=" + "\"" + str(dic["title"]) + "\""

    if "timestamp" in dic: 
        call += meta + " timestamp=" + str(dic["timestamp"]) 
    # for key in dic.keys():
    #     value = dic[key]
    #     if key == "measures":
    #         call += meta + " CHANNELS=" + str(len(value)) + meta + " CHANNEL_TAGS=\""
    #         call += ",".join(value) + "\""
    #     elif key.lower() in ["duration","encoder"]:
    #         pass
    #     else:
    #         call += meta + " " + key + "=" + "\"" + str(value) + "\""
    return call

def makeMetaArgs(title, type, measures=None, ts=None, stream=0):
    """
    Return given metadata as ffmpeg call arguments.

    Constructed metadata arguments from given title type and measures and stream.
    This arguments can be passed to ffmpegf call (e.g. as extraArgs - see other functions in this class).

    :param title:       Title added to the stream
    :type  title:       str
    :param type:        Type of the stream (audio, subtitle/subs, video etc.)
    :type  type:        str
    :param measures:    samplingrate of the given data
    :type  measures:    list
    :param stream:      potential extra arguments for ffmpeg. Consider to add ChannelNames here, default="".
    :type  stream:      int

    :return: metadata for the stream as ffmpeg arguments
    :rtype:  str
    """
    if type[0] not in ["s", "a", "v"]:
        print(bcolors.FAIL + "Currently only \"audio\", \"subtitles\" or \"video\" is allowed as type"+ bcolors.ENDC)
        return ""
    meta = " -metadata:s:" + str(type[0]) + ":" + str(stream)
    call = ""
    if ts is not None:
        call += meta + " timestamp=" + str(len(ts)) 
    if measures is not None:
        call += meta + " CHANNELS=" + str(len(measures)) + meta + " CHANNEL_TAGS=\""
        call += ",".join(measures) + "\""
    call += meta + " title=" + "\"" + title + "\""
    return call

def returnMetadataForDataDict(dataDict):
    """
    Return metadata for a given datadict.

    :param dataDict:       Title added to the stream
    :type  dataDict:       list of 

    :return: metadata for the data
    :rtype:  dict
    """
    metadata = {}
    if "measures" in dataDict: 
        metadata["CHANNELS"] = len(dataDict["measures"])
        metadata["CHANNEL_TAGS"] = "\"" + ",".join(dataDict["measures"]) + "\""
    if "title" in dataDict: metadata["title"] = dataDict["title"]
    

def makeMkv(theDataList, outFilePath, extraArgs="", verbose=False):
    """
    Make mkv from given dataList.
    theDataList needs the following structure:

    .. code-block:: python3

        theDataList = [
            {
                data:         < data as recarray with fieldnames >,
                samplingrate: < samplingrate of data >,
                type:         < type of the stream >,                 # optional otherwise set to audio
                title:        < streamTitle >,                        # optional otherwise set to Stream x
                streamIndex:  < index of the stream >,                # optional
                metadata:     < metadata of the stream >,             # optional
                measures:     < list of measures >,                   # optional
            },
            ...
        ]

    :param theDataList:     List of datadict or datadict according to scheme from mkv.load
    :type  theDataList:     list of dicts or dict
    :param outFilePath:     filepath the mkv is stored to
    :type  outFilePath:     str
    :param extraArgs:       potential extra arguments for ffmpeg. Consider to add ChannelNames here, default="".
    :type  extraArgs:       str
    """
    import subprocess
    # Convert from dict to list
    dataList = theDataList
    if isinstance(theDataList, dict): dataList = [theDataList]
    # Make temporary directory if not exist
    dirName = os.path.join(os.path.dirname(outFilePath), 'tempDir')
    if not os.path.exists(dirName): os.mkdir(dirName)
    fileNames = []
    # Loop over dictList
    for i, dataDict in enumerate(dataList):
        # Check if metatags exist
        if "title" not in dataDict: dataDict["title"] = "Stream " + str(i)
        if "type" not in dataDict: dataDict["type"] = "audio"
        measures = None
        if "measures" in dataDict: measures = dataDict["measures"]
        # Temp file name for this stream
        tmpFileName = os.path.join(dirName, "stream_" + str(i) + ".mkv")
        fileNames.append(tmpFileName)
        # Make mkv from this stream
        makeMkvRaw(dataDict["data"], tmpFileName, dataDict["samplingrate"], makeMetaArgsFromDict(dataDict, dataDict["type"]), verbose=verbose)

        if "subs" in dataDict:
            tmpSubFileName = os.path.join(dirName, "stream_" + str(i) + "_sub.ass")
            tmpSubMKVFileName = os.path.join(dirName, "stream_" + str(i) + "_sub.mkv")
            subs = SSAFile()
            for sub in dataDict["subs"]:
                subs.append(sub)
            subs.save(tmpSubFileName, encoding="utf-8")
            systemCall = "ffmpeg -hide_banner -i " + tmpSubFileName
            
            systemCall += makeMetaArgs(dataDict["title"], "subtitle")
            systemCall += " -y " + tmpSubMKVFileName
            __call_ffmpeg(systemCall, verbose=verbose)
            fileNames.append(tmpSubMKVFileName)

    # combine all streams
    combine(fileNames, outFilePath, verbose=verbose, reencode=True)
    # Remove tempdir
    subprocess.check_output(['rm', '-rf', dirName])

def makeMkvRaw(data, outFilePath, samplingRate, extraArgs="", verbose=False):
    """
    Make mkv from given data.

    :param data:            Data recorded with e.g. measurementsystem. Channels determined by #columns
    :type  data:            recarray or np.array
    :param outFilePath:     filepath the mkv is stored to
    :type  outFilePath:     str
    :param samplingRate:    samplingrate of the given data
    :type  samplingRate:    int
    :param extraArgs:       potential extra arguments for ffmpeg. Consider to add ChannelNames here, default="".
    :type  extraArgs:       str
    """
    import subprocess
    channels = len(data.dtype)
    systemCall = "ffmpeg -hide_banner -f f32le -ar " + str(samplingRate) + " -guess_layout_max 0 -ac " + str(channels) #+ " -use_wallclock_as_timestamps true"
    systemCall += " -i pipe:0 -c:a "
    # Wavpack files can only be loaded with 7 or less channels
    if channels > 7: systemCall += "pcm_f32le "
    else: systemCall += "wavpack "
    systemCall += extraArgs + " -y " + outFilePath
    if verbose:
        ffmpegProcess = subprocess.Popen(systemCall, shell=True, stdin=subprocess.PIPE, preexec_fn=os.setsid)
    else:
        ffmpegProcess = subprocess.Popen(systemCall, shell=True, stdin=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)
    # Batchsize = 1000
    # for i in range(0, len(data), Batchsize):
    #     ffmpegProcess.stdin.write(data[i:i+Batchsize].view(np.float32).reshape(data[i:i+Batchsize].shape + (-1,)).flatten().tobytes())
    ffmpegProcess.stdin.write(data.view(np.float32).reshape(data.shape + (-1,)).flatten().tobytes())
    
    ffmpegProcess.stdin.close()
    ffmpegProcess.wait()

def bakeSubtitles(inFilePath, subtitlePaths, outFilePath, titles=None, extraArgs="", verbose=False):
    """
    Merge srt or ass and an mkv files.

    :param inFilePath:        Filepath of input mkvr
    :type  inFilePath:        str
    :param subtitlePaths:     Filepath(s) of the srt or ass subtitles
    :type  subtitlePaths:     list or str
    :param outFilePath:       filepath the merged mkv is stored to
    :type  outFilePath:       str
    :param titles:            title(s) of the subtitles. They should match the stream titles inside mkv, default=None
    :type  titles:            list or str
    :param extraArgs:         potential extra arguments for ffmpeg, default=""
    :type  extraArgs:         str
    """
    systemCall = "ffmpeg -hide_banner -i " + inFilePath
    subs = subtitlePaths
    if isinstance(subtitlePaths, str):
        subs = [subtitlePaths]
    if isinstance(titles, str):
        tits = [titles]
        print(tits)
    for sub in subs:
        systemCall += " -f"
        if ".ass" in sub: systemCall += " ass"
        elif ".srt" in sub: systemCall += " srt"
        systemCall += " -i " + sub
    systemCall += " -map 0"
    for i in range(len(subs)):
        systemCall += " -map " + str(i+1)
    if titles is not None:
        if len(tits) != len(subs):
            raise AssertionError("Must specify a title for each subtitle or no title at all")
        for i, title in enumerate(tits):
            systemCall += makeMetaArgs(title, "subtitle", stream=i)
    systemCall += " -c:a copy " + extraArgs + " -y " + outFilePath
    __call_ffmpeg(systemCall, verbose=verbose)


def combine(filePaths, filePath, extraArgs="", cutToShortest=False, cutTo=None, reencode=True, verbose=False):
    """
    Combine two mkv files.

    All streams of the two mkv files will be stored in a new mkv file

    :param filePaths:        list of filepaths to combine
    :type  filePaths:        list
    :param filePath:         filepath the combined is stored to
    :type  filePath:         str
    :param extraArgs:        potential extra arguments for ffmpeg, default=""
    :type  extraArgs:        str
    :param cutToShortest:    If should be cutted to lenght of shortest stream
    :type  cutToShortest:    bool, default: ``False``
    :param cutTo:            Cut after given amount of seconds
    :type  cutTo:            float, default: ``None``
    :param reencode:         If output should be reencoded
    :type  reencode:         bool, default: ``False``
    :param verbose:          Increase output verbosity
    :type  verbose:          bool, default: ``False``
    """
    # Open the ffmpeg subprocess
    systemCall = "ffmpeg -hide_banner"
    for fp in filePaths:
        systemCall += " -i " + fp
    # All channels from each file to should be mapped to output
    for i,fp in enumerate(filePaths):
        systemCall += " -map " + str(i)
    for i,fp in enumerate(filePaths):
        systemCall += " -c:a:" + str(i)
        if cutToShortest or reencode: systemCall += " wavpack"
        else: systemCall += " copy"
    # TODO: Cut to shortest does not work, we need to rencode the data which does not seem to work atm
    if cutToShortest: systemCall += " -acodec wavpack -shortest"
    systemCall += " " + extraArgs
    if cutTo is not None: systemCall += " -t " + str(round(cutTo, 3))
    systemCall += " -y " + filePath
    # print(systemCall)
    __call_ffmpeg(systemCall, verbose=verbose)


def concat(filePaths, filePath, extraArgs="", verbose=False):
    """
    Concat given mkv files.

    All given mkv files are concatinated.
    TODO: I don't know how ffmpeg treats multiple streams here. Does it even work for multiple?

    :param filePaths:        list of filepaths to concat
    :type  filePaths:        list
    :param filePath:         filepath the concatinated data is stored to
    :type  filePath:         str
    :param extraArgs:        potential extra arguments for ffmpeg, default=""
    :type  extraArgs:        str
    """
    import subprocess
    # Create a temporary text file with the files to concat
    with open('tmp_concat.txt', 'w') as file:
        for fp in filePaths:
            file.write("file '" + str(fp) + "'\n")
    # Open the ffmpeg subprocess
    systemCall = "ffmpeg -hide_banner -f concat -safe 0 -i tmp_concat.txt -c copy " + extraArgs + " -y " + "'" + filePath + "'"
    __call_ffmpeg(systemCall, verbose=verbose)
    # Delete the temporary created concat file
    subprocess.check_output(['rm', '-rf', 'tmp_concat.txt'])


def __format_time(time, time_base):
    """
    Return time in formated way.

    :param time:       Time of interest (e.g. frame or packet pts/dts)
    :type  time:       str
    :param time_base:  Timebase of av file
    :type  time_base:  av.time_base

    :return: fomated time string
    :rtype:  str
    """
    if time is None:
        return 'None'
    return '%.3fs (%s or %s/%s)' % (time_base * time, time_base * time, av.time_base.numerator * time, av.time_base.denominator)

def getMeta(path, streamIndex=0):
    """
    Return metadata of mkv stream.

    :param path:         Filepath of mkv
    :type  path:         str
    :param streamIndex:  Index of stream which metadata is returned, default=0
    :type  streamIndex:  index

    :return: dict of metadata
    :rtype:  dict
    """
    try:
        container = av.open(path)
    except av.AVError:
        return None
    for stream in container.streams:
        if streamIndex == stream.index:
            return stream.metadata
    return None

def info(path, format=None, option=[]):
    """
    Return info of given mkv file.

    :param path:   Filepath of mkv
    :type  path:   str
    :param format: Format of the file, default=None: guessed by file ending
    :type  format: str
    :param option: Options passed to av.open(), default=[]
    :type  option: av options parameter
    """
    options = dict(x.split('=') for x in option)
    try:
        container = av.open(path, format=format, options=options)
    except av.AVError:
        return None
    info = {}
    # This extracts container info
    info["format"] = container.format
    info["duration"] = float(container.duration) / av.time_base
    info["metadata"] = container.metadata
    info["#streams"] = len(container.streams)
    info["streams"] = []
    # Getting number of samples for each stream
    samples = getSamples(path)
    # Enumerate all streams and extract stream specific info
    for i, stream in enumerate(container.streams):
        streamInfo = {}
        # Type (audio, video, subs)
        streamInfo["type"] = stream.type
        # index in of stream
        streamInfo["streamIndex"] = stream.index
        # Video and audio have a stream format
        if stream.type in ['audio', 'video']:
            streamInfo["format"] = stream.format
            # Samplingrate
            streamInfo["samplingrate"] = stream.sample_rate
            # extract # samples and duration
            if samples is not None:
                streamInfo["samples"] = samples[i]
                streamInfo["duration"] = samples[i]/streamInfo["samplingrate"]
            else:
                streamInfo["duration"] = 0
                streamInfo["samples"] = int(streamInfo["duration"]*streamInfo["samplingrate"])
        # Audio has number of channels
        if stream.type == 'audio':
            streamInfo["#channels"] = stream.channels 
            # Extract the channel tags / measures 
            channelTags = channelTags = ["C" + str(i) for i in range(stream.channels)]
            for key in ["CHANNEL_TAGS", "Channel_tags"]:
                if key in stream.metadata:
                    channelTags = stream.metadata[key].split(",")
                    break;
            streamInfo["measures"] = channelTags
        # Start time (0 for most cases)
        streamInfo["start_time"] = stream.start_time
        # Copy metadata dictionary
        streamInfo["metadata"] = stream.metadata
        # Extract stream title if there is any
        key = set(["Title", "title", "TITLE", "NAME", "Name", "name"]).intersection(set(stream.metadata.keys()))
        if len(key) > 0: title = stream.metadata[next(iter(key))]
        else: title = "Stream " + str(stream.index)
        streamInfo["title"] = title
        streamInfo["timestamp"] = 0
        # Extract timestamp if there is any
        for key in ["TIMESTAMP", "Timestamp", "timestamp"]:
            if key in stream.metadata:
                streamInfo["timestamp"] = float(stream.metadata[key])
                break
       
        info["streams"].append(streamInfo)
    # Duration of container is longest duration of all streams
    
    info["duration"] = max([info["streams"][i]["duration"] for i in range(len(container.streams)) if "duration" in info["streams"][i]])
    return info

def getSamples(path, format=None, option=[]):

    options = dict(x.split('=') for x in option)
    try:
        container = av.open(path, format=format, options=options)
    except av.AVError:
        return 0
    # all streams to be extracted
    streams = [s for s in container.streams]
    samples = [0 for _ in range(len(streams))]
    for i, stream in enumerate(streams):
        if stream.type != "audio": continue
        try:
            container = av.open(path, format=format, options=options)
        except av.AVError:
            return 0
        # Seek to the last frame in the container
        container.seek(sys.maxsize, whence='time', any_frame=False, stream=stream)
        for frame in container.decode(streams=stream.index):
            samples[i] = int(frame.pts / 1000.0*frame.rate + frame.samples)
    return samples


def printInfo(path, audio=True, video=True, subs=True, count=5, format=None, option=[]):
    """
    Display info of given mkv file.

    :param path:   Filepath of mkv
    :type  path:   str
    :param audio:  Include info about audio streams, default=True
    :type  audio:  bool
    :param video:  Include info about video streams, default=True
    :type  video:  bool
    :param subs:   Include info about subtitels, default=True
    :type  subs:   bool
    :param count:  amount of packet info is displayed
    :type  count:  int
    :param format: Format of the file, default=None: guessed by file ending
    :type  format: str
    :param option: Options passed to av.open(), default=[]
    :type  option: av options parameter
    """
    options = dict(x.split('=') for x in option)
    try:
        container = av.open(path, format=format, options=options)
    except av.AVError:
        print("Error opening file")
        return
    print('container:', container)
    print('\tformat:', container.format)
    print('\tduration:', float(container.duration) / av.time_base)
    print('\tmetadata:')
    for k, v in sorted(container.metadata.items()):
        print('\t\t%s: %r' % (k, v))
    print()

    print(len(container.streams), 'stream(s):')

    for i, stream in enumerate(container.streams):
        print('\t%r' % stream)
        print('\t\ttime_base: %r' % stream.time_base)
        if stream.type != 'subtitle':
            print('\t\trate: %r' % stream.rate)
        print('\t\tstart_time: %r' % stream.start_time)
        print('\t\tduration: %s' % __format_time(stream.duration, stream.time_base))
        print('\t\tbit_rate: %r' % stream.bit_rate)
        print('\t\tbit_rate_tolerance: %r' % stream.bit_rate_tolerance)
        if stream.type == 'audio':
            print('\t\taudio:')
            print('\t\t\tformat:', stream.format)
            print('\t\t\tchannels: %s' % stream.channels)
        elif stream.type == 'video':
            print('\t\tvideo:')
            print('\t\t\tformat:', stream.format)
            print('\t\t\taverage_rate: %r' % stream.average_rate)
        print('\t\tmetadata:')
        for k, v in sorted(stream.metadata.items()):
            print('\t\t\t%s: %r' % (k, v))
        print()

    # all streams to be extracted
    streams = [s for s in container.streams if
        (s.type == 'audio' and audio) or
        (s.type == 'video' and video) or
        (s.type == 'subtitle' and subs)
    ]

    frame_count = 0

    for i, packet in enumerate(container.demux(streams)):

        print('%02d %r' % (i, packet))
        print('\tduration: %s' % __format_time(packet.duration, packet.stream.time_base))
        print('\tpts: %s' % __format_time(packet.pts, packet.stream.time_base))
        print('\tdts: %s' % __format_time(packet.dts, packet.stream.time_base))

        for frame in packet.decode():
            frame_count += 1

            print('\tdecoded:', frame)
            print('\t\tpts:', __format_time(frame.pts, packet.stream.time_base))

            if packet.stream.type == 'video':
                pass
            elif packet.stream.type == 'audio':
                print('\t\tsamples:', frame.samples)
                print('\t\tformat:', frame.format.name)
                print('\t\tlayout:', frame.layout.name)
            elif packet.stream.type == 'subtitle':
                sub = frame
                print('\t\tformat:', sub.format)
                print('\t\tstart_display_time:', __format_time(sub.start_display_time, packet.stream.time_base))
                print('\t\tend_display_time:', __format_time(sub.end_display_time, packet.stream.time_base))
                print('\t\trects: %d' % len(sub.rects))
                for rect in sub.rects:
                    print('\t\t\t%r' % rect)
                    if rect.type == 'ass':
                        print('\t\t\t\tass: %r' % rect.ass)


            if count and frame_count >= count:
                return


def displayInfo(dataList):
    for dataDict in dataList:
        print(bcolors.OK + str(dataDict["title"]) + ":"+ bcolors.ENDC)
        print("\tSamplingrate: " + str(dataDict["samplingrate"]) + "Hz")
        print("\tMetadata: " + str(dataDict["metadata"]))
        print("\tMeasures: " + str(dataDict["measures"]))
        print("\tEntries: " + str(len(dataDict["data"])))
        print("\tLength: " + str(len(dataDict["data"])/dataDict["samplingrate"]) + "s")

def initParser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=argparse.FileType('r'),
                            help="Path to the input MKV file")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                            help="Increase output verbosity")
    return parser
    
# _______________Can be called as main__________________
if __name__ == '__main__':
    parser = initParser()
    args = parser.parse_args()

    fileName = args.input.name

    infoList = info(fileName)

    if args.verbose:
        print(infoList)

    def pretty(d, indent=0):
        for key, value in d.items():
            print('\t' * indent + str(key))
            if isinstance(value, dict):
                pretty(value, indent+1)
            else:
                print('\t' * (indent+1) + str(value))


    pretty(infoList, 0)
        
    print(("Bye Bye from " + str(os.path.basename(__file__))))
