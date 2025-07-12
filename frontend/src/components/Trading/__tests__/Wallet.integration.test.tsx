// Mock de los módulos antes de importar el componente
jest.mock('../../../contexts/AuthContext', () => ({
  useAuth: jest.fn(),
}));

jest.mock('../../../hooks/useWallet', () => ({
  useWallet: jest.fn(),
}));

import React, { act } from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import Wallet from '../Wallet';
import * as AuthContext from '../../../contexts/AuthContext';
import * as useWalletHook from '../../../hooks/useWallet';

const mockToken = 'test-token';
const mockFetchWallet = jest.fn();
const mockRecharge = jest.fn();
const mockRefreshTransactions = jest.fn();
const mockTransactions = [
  { id: 1, type: 'recharge', amount: 10000, description: 'Recarga inicial', created_at: new Date().toISOString() },
  { id: 2, type: 'trade', amount: -500, description: 'Compra simulada', created_at: new Date().toISOString() },
];

describe('Wallet integration', () => {
  beforeEach(() => {
    // Configurar mocks en beforeEach para asegurar que se apliquen antes de cada test
    (AuthContext.useAuth as jest.Mock).mockReturnValue({
      isLoading: false,
      user: null,
      subscription: null,
      login: jest.fn(),
      logout: jest.fn(),
      checkSubscription: jest.fn(),
      hasFeature: jest.fn(),
      canAccess: jest.fn(),
    });
    
    (useWalletHook.useWallet as jest.Mock).mockReturnValue({
      balance: 9500,
      transactions: mockTransactions,
      loading: false,
      error: null,
      fetchWallet: mockFetchWallet,
      recharge: mockRecharge,
      trade: jest.fn(),
      refreshTransactions: mockRefreshTransactions,
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('muestra el balance y los movimientos', () => {
    render(<Wallet />);
    expect(screen.getByText(/Balance virtual/i)).toBeInTheDocument();
    expect(screen.getByText('$9,500')).toBeInTheDocument();
    expect(screen.getByText('recharge')).toBeInTheDocument();
    expect(screen.getByText('trade')).toBeInTheDocument();
  });

  it('permite abrir el modal de recarga y recargar saldo', async () => {
    mockRecharge.mockResolvedValueOnce(true);
    render(<Wallet />);
    fireEvent.click(screen.getByText(/Añadir saldo/i));
    expect(screen.getByText(/Añadir saldo virtual/i)).toBeInTheDocument();
    fireEvent.change(screen.getByPlaceholderText(/Monto a añadir/i), { target: { value: '1000' } });
    fireEvent.click(screen.getByText(/Confirmar/i));
    await waitFor(() => {
      expect(mockRecharge).toHaveBeenCalledWith(1000);
    });
  });
}); 