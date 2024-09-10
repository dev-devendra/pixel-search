"use client";

import { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import videojs from 'video.js';
import type VideoJsPlayer from 'video.js/dist/types/player';
import 'video.js/dist/video-js.css';
import Layout from './layout';
import './styles.css';
import { track } from '@vercel/analytics';
import Head from 'next/head';

// Components
import { PhotoFrameIcon, MagnifyingGlassIcon, QuestionMarkCircleIcon } from './components/Icons';
import Footer from './components/Footer';
import { handleFileUpload } from './components/FileUploadHandler';


declare global {
  interface Window {
    webkitSpeechRecognition: any;
  }
}

// Handles Python backend API URL based on the environment
const API_URL = (() => {
  switch (process.env.NEXT_PUBLIC_VERCEL_ENV) {
    case "development":
      return process.env.NEXT_PUBLIC_DEVELOPMENT_URL || 'http://localhost:8000';
    case "preview":
      return `https://${process.env.NEXT_PUBLIC_VERCEL_URL || ''}`;
    case "production":
      return `https://${process.env.NEXT_PUBLIC_VERCEL_PROJECT_PRODUCTION_URL || ''}`;
    case "demo":
      return "";
    default:
      return "http://localhost:8000";
  }
})();

interface Result {
  score: number;
  metadata: {
    gcs_file_name: string;
    gcs_public_url: string;
    gcs_file_path: string;
    file_type: 'image' | 'video' | 'text';
    start_offset_sec: number;
    end_offset_sec: number;
    interval_sec: number;
    segment: number;
    tags: string[];
  };
}

export default function Home() {
  const [query, setQuery] = useState<string>('');
  const [results, setResults] = useState<Result[]>([]);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [dragging, setDragging] = useState<boolean>(false);
  const [totalVectors, setTotalVectors] = useState<number | null>(null);
  const [isSearchComplete, setIsSearchComplete] = useState<boolean>(false);
  const [searchTime, setSearchTime] = useState<number | null>(null);
  const [searchType, setSearchType] = useState<'text' | 'image' | 'video' | null>(null);
  const [isInputEmpty, setIsInputEmpty] = useState<boolean>(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isLoadingResults, setIsLoadingResults] = useState<boolean>(false);
  const [showSuggestions, setShowSuggestions] = useState<boolean>(false);
  // State variables to manage recording status, completion, and transcript
  const [isRecording, setIsRecording] = useState(false);
  const [recordingComplete, setRecordingComplete] = useState(false);
  const [transcript, setTranscript] = useState("");

  // Reference to store the SpeechRecognition instance
  const recognitionRef = useRef<any>(null);

  // Function to start recording
  const startRecording = () => {
    setIsRecording(true);
    // Create a new SpeechRecognition instance and configure it
    recognitionRef.current = new window.webkitSpeechRecognition();
    recognitionRef.current.continuous = true;
    recognitionRef.current.interimResults = true;

    // Event handler for speech recognition results
    recognitionRef.current.onresult = (event: any) => {
      const { transcript } = event.results[event.results.length - 1][0];

      // Log the recognition results and update the transcript state
      console.log(event.results);
      setTranscript(transcript);
    };

    // Start the speech recognition
    recognitionRef.current.start();
  };

  // Cleanup effect when the component unmounts
  useEffect(() => {
    return () => {
      // Stop the speech recognition if it's active
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  // Function to stop recording
  const stopRecording = async () => {
    if (recognitionRef.current) {
      // Stop the speech recognition and mark recording as complete
      recognitionRef.current.stop();
      setRecordingComplete(true);
      console.log("transcript:", transcript); 
      await handleAudioSubmit();
    }
  };

  // Toggle recording state and manage recording actions
  const handleToggleRecording = () => {
    setIsRecording(!isRecording);
    if (!isRecording) {
      startRecording();
    } else {
      stopRecording();
    }
  };

  const suggestions = [
    "Summer beach outfit",
    "Business casual for women",
    "Streetwear look",
    "Autumn jewelry aesthetic",
    "Minimalist wardrobe essential",
    "Athleisure outfit ideas",
    "Formal evening wear",
    "Vintage inspired look",
    "Thrift store gems",
    "Casual kpop"
  ];
  const handleSuggestionClick = (suggestion: string) => {
    setQuery(suggestion);
    setIsInputEmpty(false);
    setShowSuggestions(false);
  };

  const clearResults = () => {
    setQuery('');
    setResults([]);
    setIsInputEmpty(true);
    setIsSearchComplete(false);
    setSearchTime(null);
    setSearchType(null);
    setErrorMessage(null);
  };

  const playersRef = useRef<{ [key: string]: VideoJsPlayer }>({});

  const VerticalDivider = () => (
    <div className="h-6 w-px bg-gray-200"></div>
  );

  useEffect(() => {
    let scrollTracked = false;
    const handleScroll = () => {
      const scrollPercentage = (window.scrollY / (document.documentElement.scrollHeight - window.innerHeight)) * 100;
      if (scrollPercentage > 50 && !scrollTracked) {
        track('scroll_depth', { depth: '50%' });
        scrollTracked = true;
      }
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    const pageViewData = {
      timestamp: new Date().toISOString(),
      screenSize: `${window.screen.width}x${window.screen.height}`,
      deviceType: /Mobi|Android/i.test(navigator.userAgent) ? 'mobile' : 'desktop',
      browserName: navigator.userAgent,
      referrer: document.referrer,
      loadTime: performance.now(),
      language: navigator.language,
      totalVectors: totalVectors,
      appVersion: process.env.NEXT_PUBLIC_APP_VERSION || 'unknown',
    };

    track('page_viewed', pageViewData);
  }, [totalVectors]);

  useEffect(() => {

    const fetchTotalVectors = async () => {
      try {
        const response = await axios.get(`${API_URL}/api/index/info`);
        setTotalVectors(response.data.total_vectors);
      } catch (error) {
        console.error('Error fetching total vectors:', error);
      }
    };

    fetchTotalVectors();
  }, []);

  useEffect(() => {
    return () => {
      Object.values(playersRef.current).forEach(player => {
        if (player && typeof player.dispose === 'function') {
          player.dispose();
        }
      });
      playersRef.current = {};
    };
  }, []);

  useEffect(() => {
    results.forEach((result, index) => {
      if (result.metadata.file_type === 'video') {
        const videoId = getVideoId(result, index);
        const videoElement = document.getElementById(videoId) as HTMLVideoElement;

        if (videoElement && !playersRef.current[videoId]) {
          const player = videojs(videoElement, {
            aspectRatio: '1:1',
            fluid: true,
            controls: true,
            muted: true,
            preload: 'auto'
          });

          player.one('ready', () => {
            player.currentTime(result.metadata.start_offset_sec);
          });

          playersRef.current[videoId] = player;
        }
      }
    });

    return () => {
      Object.keys(playersRef.current).forEach(videoId => {
        if (!results.some((result, index) => getVideoId(result, index) === videoId)) {
          playersRef.current[videoId].dispose();
          delete playersRef.current[videoId];
        }
      });
    };
  }, [results]);

  const resetSearchState = () => {
    setResults([]);
    setIsSearchComplete(false);
    setSearchTime(null);
    setSearchType(null);
    setErrorMessage(null);
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (isInputEmpty) return;

    setShowSuggestions(false);

    resetSearchState();

    setIsSearching(true);
    setIsSearchComplete(false);
    setSearchTime(null);
    setSearchType('text');
    setErrorMessage(null);
    setIsLoadingResults(true);
    const startTime = Date.now();
    try {
      const response = await axios.post(`${API_URL}/api/search/text`, { query });
      setResults(response.data.results);
      const endTime = Date.now();
      setSearchTime(endTime - startTime);
      setIsSearchComplete(true);
      track('search_results', {
        searchType,
        query,
        searchTime
      });
    } catch (error) {
      console.error('Error during text search:', error);
      if (axios.isAxiosError(error) && error.response) {
        setErrorMessage(`Oops! ${error.response.data.detail || 'An unexpected error occurred'}`);
      } else {
        setErrorMessage('Oops! An unexpected error occurred. Our engineers have been notified.');
      }
    } finally {
      setIsSearching(false);
      setIsLoadingResults(false);
    }
  };

  const handleAudioSubmit = async () => {
    if (!transcript || transcript.trim() === "") return; // Check if transcript is empty
  
    setShowSuggestions(false);
  
    resetSearchState();
  
    setIsSearching(true);
    setIsSearchComplete(false);
    setSearchTime(null);
    setSearchType('text');
    setErrorMessage(null);
    setIsLoadingResults(true);
    const startTime = Date.now();
  
    try {
      // Using transcript as the value for the search instead of query
      const response = await axios.post(`${API_URL}/api/search/text`, { query: transcript });
      setResults(response.data.results);
      const endTime = Date.now();
      setSearchTime(endTime - startTime);
      setIsSearchComplete(true);
      track('search_results', {
        searchType: 'audio', // Mark search type as 'audio' for tracking
        query: transcript,   // Use transcript for tracking
        searchTime: endTime - startTime,
      });
    } catch (error) {
      console.error('Error during Audio search:', error);
      if (axios.isAxiosError(error) && error.response) {
        setErrorMessage(`Oops! ${error.response.data.detail || 'An unexpected error occurred'}`);
      } else {
        setErrorMessage('Oops! An unexpected error occurred. Our engineers have been notified.');
      }
    } finally {
      setIsSearching(false);
      setIsLoadingResults(false);
    }
  };
  
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setQuery(value);
    setIsInputEmpty(value.trim() === '');
    setShowSuggestions(value.trim() !== '');
  };

  const handleFileUploadWrapper = async (file: File) => {
    resetSearchState();
    await handleFileUpload(file, {
      API_URL,
      setSearchType,
      setErrorMessage,
      setIsUploading,
      setIsSearchComplete,
      setSearchTime,
      setIsLoadingResults,
      setResults,
      setQuery,
      setIsInputEmpty,
      setIsSearching
    });
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      track('file_selected');
      await handleFileUploadWrapper(e.target.files[0]);
    }
  };

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
  }, []);

  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      track('file_dropped');
      await handleFileUploadWrapper(e.dataTransfer.files[0]);
    }
  };

  const getScoreLabel = (score: number) => {
    return { score: score.toFixed(4) };
  };
  const getTagsList = (tags: string[]) => {
    return { tags: tags.join(", ") };
  };
  const getVideoId = (result: Result, index: number) => `video-${index}-${result.metadata.gcs_public_url}`;

  return (
    <Layout>
      <Head>
        <title>Pixel-Search</title>
      </Head>
      <div
        className={`flex flex-col items-center justify-start min-h-screen bg-gray-100 ${dragging ? 'border-4 border-dashed border-blue-500' : ''
          }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        style={{ fontFamily: "'Inter', 'Helvetica', 'Arial', sans-serif" }}
      >
        <div className="max-w-6xl w-full px-4 md:px-0 mt-12">
        <h1 className="font-sans text-4xl mb-3 text-center text-teal-700">Pixel-Search</h1>
          <h1 className="font-sans text-base mb-5 text-center text-gray-900">Upload a photo or video or search by text</h1>
          <div className="max-w-xl mx-auto relative">
            <form onSubmit={handleSubmit} className="flex items-center">
              <div className="flex-grow flex items-center bg-white rounded shadow-md">
                <div className="flex-grow relative">
                  <input
                    type="text"
                    value={query}
                    onChange={handleInputChange}
                    onFocus={() => setShowSuggestions(true)}
                    onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
                    placeholder="Describe the outfit, or drag in an image or video"
                    className="w-full flex-grow px-6 py-3 text-gray-700 bg-transparent focus:outline-none"
                    disabled={isUploading || isSearching}
                  />
                  {showSuggestions && suggestions.length > 0 && (
                    <div className="absolute left-0 right-0 mt-2 w-full bg-white border rounded shadow-lg max-h-60 overflow-y-auto z-10" style={{ width: 'calc(100% + 59px)', marginLeft: '-1px' }}>
                      <div className="mt-3"></div>
                      {suggestions.map((suggestion, index) => (
                        <div
                          key={index}
                          className="px-6 py-1.5 hover:bg-gray-100 cursor-pointer text-gray-700  flex items-center"
                          onClick={() => handleSuggestionClick(suggestion)}
                        >
                          <MagnifyingGlassIcon className="h-4 w-4 mr-3 text-indigo-500" />
                          {suggestion}
                        </div>
                      ))}
                    </div>
                  )}
                  </div>  
                  <input
                    type="file"
                    accept="image/*,video/*"
                    onChange={handleFileChange}
                    style={{ display: 'none' }}
                    id="upload-input"
                    disabled={isUploading || isSearching}
                  />
                  {!isInputEmpty && (
                    <>
                      <button
                        type="button"
                        onClick={clearResults}
                        className="text-gray-400 hover:text-gray-500 mr-0.5 focus:outline-none"
                        disabled={isUploading || isSearching}
                      >
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          width="44"
                          height="44"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="1"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <line x1="15" y1="9" x2="9" y2="15" />
                          <line x1="9" y1="9" x2="15" y2="15" />
                        </svg>
                      </button>
                      <VerticalDivider />
                    </>
                  )}
                <label htmlFor="upload-input" className={`cursor-pointer px-4 ${isUploading || isSearching ? 'text-gray-400' : 'text-gray-500 hover:text-gray-700'} focus:outline-none`}>
                  <PhotoFrameIcon className="h-6 w-6" />
                </label>
              </div>
              <button
                type="submit"
                className={`ml-1 px-3 ${isInputEmpty
                    ? 'text-gray-400 cursor-not-allowed'
                    : isUploading || isSearching
                      ? 'text-gray-400 cursor-wait'
                      : 'text-gray-500 hover:text-gray-700'
                  } focus:outline-none`}
                disabled={isInputEmpty || isUploading || isSearching}
              >
                <MagnifyingGlassIcon className="h-6 w-6 text-indigo-500 hover:text-indigo-700" />
              </button>
            </form>


            {errorMessage && (
              <div className="w-full mt-4 text-red-500 text-center">
                {errorMessage}
              </div>
            )}
            {(isUploading || isSearching) && (
              <div className="w-full mt-8 flex items-center justify-center">
                <span className="text-gray-500 pulse">
                  {isUploading ? "Uploading, embedding, and searching..." : "Searching..."}
                </span>
                <div className="ml-3 spinner border-4 border-t-transparent border-indigo-300 rounded-full w-6 h-6 animate-spin"></div>
              </div>
            )}
          </div>
          <div className="w-full">
  

  <div className="flex items-center w-full">
    {isRecording ? (
      // Button for stopping recording
      <button
        onClick={handleToggleRecording}
        className="mt-5 m-auto flex items-center justify-center bg-red-400 hover:bg-red-500 rounded-full w-10 h-10 focus:outline-none"
      >
        <svg
          className="h-8 w-8" // Reduced size of the stop icon
          viewBox="0 0 24 24"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path fill="white" d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
        </svg>
      </button>
    ) : (
      // Button for starting recording
      <button
        onClick={handleToggleRecording}
        className="mt-5 m-auto flex items-center justify-center bg-blue-400 hover:bg-blue-500 rounded-full w-10 h-10 focus:outline-none"
      >
        <svg
          viewBox="0 0 256 256"
          xmlns="http://www.w3.org/2000/svg"
          className="w-8 h-8 text-white" // Reduced size of the mic icon
        >
          <path
            fill="currentColor"
            d="M128 176a48.05 48.05 0 0 0 48-48V64a48 48 0 0 0-96 0v64a48.05 48.05 0 0 0 48 48ZM96 64a32 32 0 0 1 64 0v64a32 32 0 0 1-64 0Zm40 143.6V232a8 8 0 0 1-16 0v-24.4A80.11 80.11 0 0 1 48 128a8 8 0 0 1 16 0a64 64 0 0 0 128 0a8 8 0 0 1 16 0a80.11 80.11 0 0 1-72 79.6Z"
          />
        </svg>
      </button>
    )}
  </div>
  {(isRecording || transcript) && (
    <div className="w-1/4 m-auto rounded-md border p-4 bg-white">
      <div className="flex-1 flex w-full justify-between">
        {isRecording && (
          <div className="rounded-full w-4 h-4 bg-red-400 animate-pulse" />
        )}
      </div>

      {transcript && (
        <div className="border rounded-md p-2 mt-4">
          <p className="mb-0">{transcript}</p>
        </div>
      )}
    </div>
  )}
</div>
          <div>
            {isSearchComplete && searchTime !== null && totalVectors !== null && (
              <div className="ml-1 mt-6 mb-2 flex items-center text-left text-gray-700">
                <p>
                  Searched {totalVectors.toLocaleString()} styles
                  {searchType === 'text' && <> for <strong className="text-indigo-800">{query}</strong></>}
                  {searchType === 'image' && <> for <strong className="text-indigo-800">your image</strong></>}
                  {searchType === 'video' && <> for <strong className="text-indigo-800">your video</strong></>}
                </p>
                <button
                  type="button"
                  onClick={clearResults}
                  className="text-gray-400 hover:text-gray-500 mb-0.4 ml-2 focus:outline-none"
                  disabled={isUploading || isSearching}
                >

                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <circle cx="12" cy="12" r="10" />
                    <line x1="15" y1="9" x2="9" y2="15" />
                    <line x1="9" y1="9" x2="15" y2="15" />
                  </svg>
                </button>
              </div>
            )}
            {isLoadingResults && (
              <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {[...Array(20)].map((_, index) => (
                  <div key={index} className="animate-pulse">
                    <div className="bg-gray-300 h-64 w-full rounded-sm"></div>
                    <div className="h-4 bg-gray-300 rounded w-3/4 mt-2"></div>
                  </div>
                ))}
              </div>
            )}
            {!isLoadingResults && results.length > 0 && (
              <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {results.map((result, index) => {
                  const { score } = getScoreLabel(result.score);
                  const videoId = getVideoId(result, index);
                  const { tags } = getTagsList(result.metadata.tags)
                  return (
                    <div key={videoId}>
                      {result.metadata.file_type === 'image' ? (
                        <img src={result.metadata.gcs_public_url} alt="Result" className="w-full h-auto object-cover mt-2 rounded hover-shadow" />
                      ) : (
                        <div className="video-container mt-2 rounded hover-shadow">
                          <video
                            id={videoId}
                            className="video-js vjs-default"
                          >
                            <source src={result.metadata.gcs_public_url} type="video/mp4" />
                            Your browser does not support the video tag.
                          </video>
                        </div>
                      )}
                      <div className="inline-block mt-2 mb-2 px-1 py-1 text-sm text-gray-400 flex items-center">
                        Similarity score: {score} Tags: {tags} 
                        <div className="relative ml-1 group">
                          <QuestionMarkCircleIcon className="h-4 w-4 text-gray-400" />
                          <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 bg-gray-500 text-white text-xs rounded py-1 px-2 hidden group-hover:block whitespace-nowrap">
                            Cosine similarity score between 0 - 1, higher is more similar. 
                            <a href="" target="_blank" rel="noopener noreferrer" className="text-blue-300 hover:text-blue-200"> About vector similarity.</a>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
        <Footer />
      </div>
    </Layout>
  );
}