import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const Signup = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({ 
    name: '', 
    email: '', 
    password: '',
    confirmPassword: ''
  });
  const [loading, setLoading] = useState(false);
  const [otpSent, setOtpSent] = useState(false);
  const [otp, setOtp] = useState('');
  const [errors, setErrors] = useState({});
  const [successMessage, setSuccessMessage] = useState('');
  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.name.trim()) {
      newErrors.name = 'Name is required';
    }
    
    if (!formData.email) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Email is invalid';
    }
    
    if (!formData.password) {
      newErrors.password = 'Password is required';
    } else if (formData.password.length < 6) {
      newErrors.password = 'Password must be at least 6 characters';
    }
    
    if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
    
    // Clear error when user starts typing in the field
    if (errors[e.target.name]) {
      setErrors({ ...errors, [e.target.name]: '' });
    }
  };

  const handleOtpChange = (e) => {
    setOtp(e.target.value);
    if (errors.otp) {
      setErrors({ ...errors, otp: '' });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) return;
    
    setLoading(true);

    try {
      const response = await axios.post(`https://stock-flipr-backend.onrender.com/api/auth/signup`, {
        name: formData.name,
        email: formData.email,
        password: formData.password
      });
      
      localStorage.setItem('email', formData.email);
      setSuccessMessage('OTP sent to your email. Please verify to continue.');
      setOtpSent(true);
      
      // Instead of immediately navigating, show success message first
      setTimeout(() => {
        navigate('/verify-otp');
      }, 2000);
      
    } catch (error) {
      const errorMsg = error.response?.data?.message || 'Something went wrong, please try again.';
      setErrors({ ...errors, submit: errorMsg });
    } finally {
      setLoading(false);
    }
  };

  const handleOtpSubmit = async (e) => {
    e.preventDefault();
    
    if (!otp) {
      setErrors({ ...errors, otp: 'OTP is required' });
      return;
    }
    
    setLoading(true);

    try {
      const email = localStorage.getItem('email');
      const response = await axios.post(`${API_URL}/api/auth/verify-otp`, { otp, email });
      
      setSuccessMessage('Account verified successfully! Redirecting to login...');
      
      setTimeout(() => {
        navigate('/login');
      }, 2000);
      
    } catch (error) {
      const errorMsg = error.response?.data?.message || 'Invalid or expired OTP. Please try again.';
      setErrors({ ...errors, otp: errorMsg });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex justify-center items-center min-h-screen dark:bg-darkBackground bg-lightBackground px-4">
      <div className="dark:bg-gray-800 bg-white p-8 rounded-lg shadow-lg w-full max-w-md">
        {successMessage && (
          <div className="mb-4 p-3 bg-green-100 border border-green-400 text-green-700 rounded">
            <div className="flex items-center">
              <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              <span>{successMessage}</span>
            </div>
          </div>
        )}
        
        {errors.submit && (
          <div className="mb-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
            <div className="flex items-center">
              <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <span>{errors.submit}</span>
            </div>
          </div>
        )}
        
        <form
          onSubmit={otpSent ? handleOtpSubmit : handleSubmit}
          className="space-y-5"
        >
          <h2 className="text-2xl font-bold text-center dark:text-blue-400 text-blue-600 mb-6">
            {otpSent ? 'Verify Your Email' : 'Create an Account'}
          </h2>

          {!otpSent ? (
            <>
              <div>
                <label htmlFor="name" className="block text-sm font-medium mb-1 dark:text-gray-300 text-gray-700">
                  Full Name
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <svg className="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <input
                    id="name"
                    type="text"
                    name="name"
                    value={formData.name}
                    placeholder="John Doe"
                    onChange={handleChange}
                    className={`w-full pl-10 pr-3 py-2 border ${errors.name ? 'border-red-500' : 'dark:border-gray-600 border-gray-300'} rounded-lg 
                            dark:bg-gray-700 dark:text-white
                            focus:outline-none focus:ring-2 focus:ring-blue-500`}
                  />
                </div>
                {errors.name && <p className="mt-1 text-sm text-red-500">{errors.name}</p>}
              </div>
              
              <div>
                <label htmlFor="email" className="block text-sm font-medium mb-1 dark:text-gray-300 text-gray-700">
                  Email Address
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <svg className="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path d="M2.003 5.884L10 9.882l7.997-3.998A2 2 0 0016 4H4a2 2 0 00-1.997 1.884z" />
                      <path d="M18 8.118l-8 4-8-4V14a2 2 0 002 2h12a2 2 0 002-2V8.118z" />
                    </svg>
                  </div>
                  <input
                    id="email"
                    type="email"
                    name="email"
                    value={formData.email}
                    placeholder="you@example.com"
                    onChange={handleChange}
                    className={`w-full pl-10 pr-3 py-2 border ${errors.email ? 'border-red-500' : 'dark:border-gray-600 border-gray-300'} rounded-lg 
                            dark:bg-gray-700 dark:text-white
                            focus:outline-none focus:ring-2 focus:ring-blue-500`}
                  />
                </div>
                {errors.email && <p className="mt-1 text-sm text-red-500">{errors.email}</p>}
              </div>
              
              <div>
                <label htmlFor="password" className="block text-sm font-medium mb-1 dark:text-gray-300 text-gray-700">
                  Password
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <svg className="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <input
                    id="password"
                    type="password"
                    name="password"
                    value={formData.password}
                    placeholder="••••••••"
                    onChange={handleChange}
                    className={`w-full pl-10 pr-3 py-2 border ${errors.password ? 'border-red-500' : 'dark:border-gray-600 border-gray-300'} rounded-lg 
                            dark:bg-gray-700 dark:text-white
                            focus:outline-none focus:ring-2 focus:ring-blue-500`}
                  />
                </div>
                {errors.password && <p className="mt-1 text-sm text-red-500">{errors.password}</p>}
              </div>
              
              <div>
                <label htmlFor="confirmPassword" className="block text-sm font-medium mb-1 dark:text-gray-300 text-gray-700">
                  Confirm Password
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <svg className="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <input
                    id="confirmPassword"
                    type="password"
                    name="confirmPassword"
                    value={formData.confirmPassword}
                    placeholder="••••••••"
                    onChange={handleChange}
                    className={`w-full pl-10 pr-3 py-2 border ${errors.confirmPassword ? 'border-red-500' : 'dark:border-gray-600 border-gray-300'} rounded-lg 
                            dark:bg-gray-700 dark:text-white
                            focus:outline-none focus:ring-2 focus:ring-blue-500`}
                  />
                </div>
                {errors.confirmPassword && <p className="mt-1 text-sm text-red-500">{errors.confirmPassword}</p>}
              </div>
              
              <button
                type="submit"
                className="w-full flex justify-center items-center bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 
                         dark:bg-blue-500 dark:hover:bg-blue-600
                         transition duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-2 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Creating Account...
                  </>
                ) : (
                  'Sign Up'
                )}
              </button>
            </>
          ) : (
            <>
              <div>
                <label htmlFor="otp" className="block text-sm font-medium mb-1 dark:text-gray-300 text-gray-700">
                  Enter OTP Code
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <svg className="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M10 2a1 1 0 011 1v1.323l3.954 1.582 1.599-.8a1 1 0 01.894 1.79l-1.233.616 1.738 5.42a1 1 0 01-.285 1.05A3.989 3.989 0 0115 15a3.989 3.989 0 01-2.667-1.019 1 1 0 01-.285-1.05l1.715-5.349L11 6.477V16h2a1 1 0 110 2H7a1 1 0 110-2h2V6.477L6.237 7.582l1.715 5.349a1 1 0 01-.285 1.05A3.989 3.989 0 015 15a3.989 3.989 0 01-2.667-1.019 1 1 0 01-.285-1.05l1.738-5.42-1.233-.617a1 1 0 01.894-1.788l1.599.799L9 4.323V3a1 1 0 011-1zm-5 8.274l-.818 2.552c.25.112.526.174.818.174.292 0 .569-.062.818-.174L5 10.274zm10 0l-.818 2.552c.25.112.526.174.818.174.292 0 .569-.062.818-.174L15 10.274z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <input
                    id="otp"
                    type="text"
                    value={otp}
                    onChange={handleOtpChange}
                    placeholder="Enter 6-digit OTP"
                    className={`w-full pl-10 pr-3 py-2 border ${errors.otp ? 'border-red-500' : 'dark:border-gray-600 border-gray-300'} rounded-lg 
                             dark:bg-gray-700 dark:text-white
                             focus:outline-none focus:ring-2 focus:ring-blue-500`}
                    maxLength={6}
                  />
                </div>
                {errors.otp && <p className="mt-1 text-sm text-red-500">{errors.otp}</p>}
                
                <p className="mt-3 text-sm dark:text-gray-400 text-gray-600">
                  OTP has been sent to your email address. The code will expire in 15 minutes.
                </p>
              </div>
              
              <button
                type="submit"
                className="w-full flex justify-center items-center bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 
                         dark:bg-blue-500 dark:hover:bg-blue-600
                         transition duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-2 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Verifying...
                  </>
                ) : (
                  'Verify OTP'
                )}
              </button>
            </>
          )}
          
          <p className="text-center text-sm dark:text-gray-400 text-gray-600">
            Already have an account?{' '}
            <a 
              href="/login" 
              className="font-medium text-blue-600 dark:text-blue-400 hover:underline"
            >
              Log in
            </a>
          </p>
        </form>
      </div>
    </div>
  );
};

export default Signup;