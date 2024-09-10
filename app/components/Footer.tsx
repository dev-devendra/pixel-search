import React from 'react';

const Footer: React.FC = () => {
  return (
    <div className="w-full mt-auto py-4 text-center text-gray-500 text-sm">
      <p>
        Built with{' '}
        <a href="" target="_blank" rel="noopener noreferrer" className="text-indigo-800 hover:underline">
          Pinecone
        </a>
        ,{' '}
        <a href="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-multimodal-embeddings" target="_blank" rel="noopener noreferrer" className="text-indigo-800 hover:underline">
          CLIP VIT H/14 Model
        </a>
        , and{' '}
        <a href="https://www.pexels.com/" target="_blank" rel="noopener noreferrer" className="text-indigo-800 hover:underline">
          Pexels
        </a>
      </p>
    </div>
  );
};

export default Footer;