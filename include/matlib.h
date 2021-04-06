#ifndef INCLUDE_MATLIB_H_
#define INCLUDE_MATLIB_H_

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <iomanip>
#include <numeric>
#include <ostream>
#include <utility>

namespace matlib
{
// operator[] for a MemoryType for type T returns reference to T
template <typename MemT, typename T>
concept MemoryType = requires (MemT m, std::ptrdiff_t i) {
    { m[i] } noexcept -> std::same_as<T&>;
};

// Represents a Matrix with compile-time memory layout
template <size_t num_rows, size_t num_columns, typename T,
          MemoryType<T> MemT = std::array<T, num_rows * num_columns>,
          std::ptrdiff_t row_stride = num_columns,
          std::ptrdiff_t column_stride = 1>
class Matrix;

// Matrix with a pointer to T as backing storage type
template <size_t num_rows, size_t num_columns, typename T, std::ptrdiff_t row_stride, std::ptrdiff_t column_stride>
using MatrixView = Matrix<num_rows, num_columns, T, T*, row_stride, column_stride>;

// Helper template succeeds substitution for Matrix types
template <size_t num_rows, size_t num_columns, typename T, MemoryType<T> MemT, std::ptrdiff_t row_stride,
          std::ptrdiff_t column_stride>
constexpr bool is_matrix(const Matrix<num_rows, num_columns, T, MemT, row_stride, column_stride>&) noexcept
{
    return true;
}

// Any template specialisation of Matrix
template <typename T>
concept MatrixType = requires (T t) { is_matrix(t); };

// Any template specialisation of Matrix with N rows
template <typename T, size_t N>
concept MatrixTypeN = requires (T t) { is_matrix<N>(t); };

// Any template specialisation of Matrix with N rows and M columns
template <typename T, size_t N, size_t M>
concept MatrixTypeNxM = requires (T t) { is_matrix<N, M>(t); };

template <typename T>
concept SquareMatrixType = MatrixTypeN<T, std::remove_reference_t<T>::k_num_columns>;

template <typename T, size_t N>
concept SquareMatrixTypeN = MatrixTypeNxM<T, N, N>;

template <typename T>
concept RowVectorType = MatrixTypeN<T, 1>;

template <typename T, size_t N>
concept RowVectorTypeN = MatrixTypeNxM<T, 1, N>;

template <typename T>
concept ColumnVectorType = MatrixTypeNxM<T, std::remove_reference_t<T>::k_num_rows, 1>;

template <typename T, size_t N>
concept ColumnVectorTypeN = MatrixTypeNxM<T, N, 1>;

template <typename T>
concept VectorType = ColumnVectorType<T>;

template <typename T, size_t N>
concept VectorTypeN = ColumnVectorTypeN<T, N>;

template <size_t num_rows_, size_t num_columns_, typename T_, MemoryType<T_> MemT_, std::ptrdiff_t row_stride_,
          std::ptrdiff_t column_stride_>
class Matrix {
public:
    static constexpr size_t k_num_rows    { num_rows_ };
    static constexpr size_t k_num_columns { num_columns_ };

    using T    = T_;
    using MemT = MemT_;

    static constexpr std::ptrdiff_t k_row_stride  { row_stride_ };
    static constexpr std::ptrdiff_t k_column_stride  { column_stride_ };

    static constexpr bool k_is_contiguous = ((k_num_rows == 1) || (k_row_stride == k_num_columns)) &&
                                            ((k_num_columns == 1) || (k_column_stride == 1));

    static constexpr size_t k_size = k_num_rows * k_num_columns;

    using Row      = MatrixView<1, k_num_columns, T, k_row_stride, k_column_stride>;
    using ConstRow = MatrixView<1, k_num_columns, const T, k_row_stride, k_column_stride>;

    using Column      = MatrixView<k_num_rows, 1, T, k_row_stride, k_column_stride>;
    using ConstColumn = MatrixView<k_num_rows, 1, const T, k_row_stride, k_column_stride>;

    template <typename U, std::ptrdiff_t u_row_stride = k_row_stride, std::ptrdiff_t u_column_stride = k_column_stride>
    using MatrixViewT = MatrixView<k_num_rows, k_num_columns, U, u_row_stride, u_column_stride>;

    // Assignment operator returns a proxy object with an operator, that can be called to sequentially initialize
    // matrix data
    class Initializer {
    public:
        constexpr Initializer(Matrix& matrix) noexcept
            : m_matrix(matrix)
        {}

        // TODO
        constexpr Initializer& operator,(const T value) noexcept
        {
            m_matrix[m_assign_pos] = value;
            ++m_assign_pos;
            m_assign_pos %= k_size;
            return *this;
        }

    private:
        Matrix& m_matrix;
        size_t m_assign_pos = {};
    };

    // Iterator for memory layout different to the 2D-indexing row-major layout
    template <typename CvT>
    struct IteratorIJ {
        using iterator_category = std::random_access_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using value_type        = T;
        using pointer           = CvT*;
        using reference         = CvT&;

        mutable MatrixView<k_num_rows, k_num_columns, CvT, k_row_stride, k_column_stride> m_matrix;
        size_t m_pos {};

        // comparison
        friend constexpr auto operator<=>(const IteratorIJ lhs, const IteratorIJ rhs) noexcept
        {
            return lhs.m_pos <=> rhs.m_pos;
        }
        friend constexpr bool operator==(const IteratorIJ lhs, const IteratorIJ rhs) noexcept
        {
            return lhs.m_pos == rhs.m_pos;
        }

        // increment operators
        constexpr IteratorIJ& operator++() noexcept { ++m_pos; return *this; }
        constexpr IteratorIJ operator++(int) const noexcept { return ++IteratorIJ{ *this }; }
        constexpr IteratorIJ& operator+=(const std::ptrdiff_t n)  noexcept{ m_pos += n; return *this; }
        friend constexpr IteratorIJ operator+(IteratorIJ it, const std::ptrdiff_t n) noexcept { return it += n; }
        friend constexpr IteratorIJ operator+(const std::ptrdiff_t n, IteratorIJ it) noexcept { return it += n; }

        // decrement operators
        constexpr IteratorIJ& operator--() noexcept { --m_pos; return *this; }
        constexpr IteratorIJ operator--(int) const noexcept { return --IteratorIJ{ *this }; }
        constexpr IteratorIJ& operator-=(const std::ptrdiff_t n) noexcept { m_pos -= n; return *this; }
        friend constexpr IteratorIJ operator-(IteratorIJ it, const std::ptrdiff_t n) noexcept { return it -= n; }
        friend constexpr IteratorIJ operator-(const std::ptrdiff_t n, IteratorIJ it) noexcept { return it -= n; }

        // dereference operators
        constexpr reference operator*() const noexcept
        {
            return m_matrix(m_pos / k_num_columns, m_pos % k_num_columns);
        }
        constexpr pointer operator->() const noexcept { return &**this; }
        constexpr reference operator[](const std::ptrdiff_t i) const noexcept { return *(*this + i); }

        // iterator distance
        friend constexpr std::ptrdiff_t operator-(const IteratorIJ lhs, const IteratorIJ rhs) noexcept
        {
            return static_cast<std::ptrdiff_t>(lhs.m_pos) - rhs.m_pos;
        }
    };

    using Iterator      = std::conditional_t<k_is_contiguous, T*, IteratorIJ<T>>;
    using ConstIterator = std::conditional_t<k_is_contiguous, const T*, IteratorIJ<const T>>;

    // Force explicit initialization of data pointer if that's the backing storage type
    constexpr Matrix() noexcept requires (!std::is_pointer_v<MemT>) = default;

    // Initialization from copy of memory type
    constexpr Matrix(const MemT& data) noexcept requires std::copy_constructible<MemT>
        : m_data(data)
    {}

    // Initialisation from any other matrix type with matching dimensions
    template <typename U, std::ptrdiff_t u_row_stride, std::ptrdiff_t u_column_stride>
    constexpr Matrix(const MatrixViewT<U, u_row_stride, u_column_stride> other) noexcept
    {
        *this = other;
    }

    // Assignment from any other matrix type with matching dimensions
    template <typename U, std::ptrdiff_t u_row_stride, std::ptrdiff_t u_column_stride>
    constexpr Matrix& operator=(const MatrixViewT<U, u_row_stride, u_column_stride> rhs) noexcept
    {
        std::copy(rhs.begin(), rhs.end(), begin());
        return *this;
    }

    // TODO
    constexpr Initializer operator=(const T value) noexcept
    {
        return Initializer{ *this }, value;
    }

    // Access element at row i, column j
    constexpr auto& operator()(const size_t i, const size_t j) noexcept { return m_data[idx(i, j)]; }
    constexpr const auto& operator()(const size_t i, const size_t j) const noexcept { return m_data[idx(i, j)]; }

    // Begin row-major iteration
    constexpr auto begin() noexcept { return Iterator{ &m_data[0] }; }
    constexpr auto cbegin() const noexcept { return ConstIterator{ &m_data[0] }; }
    constexpr auto begin() const noexcept { return cbegin(); }

    // End row-major iteration
    constexpr auto end() noexcept { return Iterator{ &m_data[0] } + k_size; }
    constexpr auto cend() const noexcept { return ConstIterator{ &m_data[0] } + k_size; }
    constexpr auto end() const noexcept { return cend(); }

    // Access element at linear index i in row-major order
    constexpr auto& operator[](const size_t i) noexcept { return begin()[i]; }
    constexpr auto& operator[](const size_t i) const noexcept { return begin()[i]; }

    // Explicit decay to MatrixView type
    constexpr auto view() noexcept { return MatrixViewT<T>{ &m_data[0] }; }
    constexpr auto cview() const noexcept { return MatrixViewT<const T>{ &m_data[0] }; }
    constexpr auto view() const noexcept { return cview(); }

    // MatrixView access to row i
    constexpr auto row(const size_t i) noexcept { return Row{ &(*this)(i, 0) }; }
    constexpr auto crow(const size_t i) const noexcept { return ConstRow{ &(*this)(i, 0) }; }
    constexpr auto row(const size_t i) const noexcept { return crow(i); }

    // MatrixView access to colummn i
    constexpr auto column(const size_t i) noexcept { return Column{ &(*this)(0, i) }; }
    constexpr auto ccolumn(const size_t i) const noexcept { return ConstColumn{ &(*this)(0, i) }; }
    constexpr auto column(const size_t i) const noexcept { return ccolumn(i); }

    // Implicit conversion to MatrixView
    constexpr operator MatrixViewT<T> () noexcept { return view(); }
    constexpr operator MatrixViewT<const T> () const noexcept { return view(); }

private:
    // 2D-row-major indexing to linear, backing storage index
    static constexpr size_t idx(const size_t i, const size_t j) noexcept
    {
        return (i * k_row_stride) + (j * k_column_stride);
    }

    MemT m_data {};
};

template <size_t num_rows, typename T>
using Vector = Matrix<num_rows, 1, T>;

// Print operation for MatrixView types
template <size_t num_rows, size_t num_columns, typename T, std::ptrdiff_t row_stride, std::ptrdiff_t column_stride>
constexpr std::ostream& operator<<(
    std::ostream& os, const MatrixView<num_rows, num_columns, const T, row_stride, column_stride> m)
{
    const auto savedFlags = os.flags();

    os << std::fixed << std::setprecision(1);

    for (size_t i = 0; i < m.k_num_rows; ++i) {
        for (size_t j = 0; j < m.k_num_columns; ++j) {
            os << +m(i, j);
            if (j != (m.k_num_columns - 1)) {
                os << ", ";
            }
        }
        if (i != (m.k_num_rows - 1)) {
            os << ",\n";
        }
    }

    os.flags(savedFlags);

    return os;
}
// Print operation for other Matrix types explicitly decay to MatrixView
constexpr std::ostream& operator<<(std::ostream& os, const MatrixType auto& m) { return os << m.cview(); }

// Transpose on 1x1 Matrix is a no-op. Simply decay to MatrixView
constexpr auto transpose(const SquareMatrixTypeN<1> auto& m) noexcept
{
    return m.view();
}

// General transpose as a const MatrixView with transposed static memory layout
template <MatrixType MRef> requires (!SquareMatrixTypeN<MRef, 1>)
constexpr auto transpose(MRef&& m) noexcept
{
    using M = std::remove_cvref_t<MRef>;
    using TransposedConstView =
        MatrixView<M::k_num_columns, M::k_num_rows, const typename M::T, M::k_column_stride, M::k_row_stride>;
    return TransposedConstView{ &m[0] };
}

// Negation operation for MatrixView types
template <size_t num_rows, size_t num_columns, typename T, std::ptrdiff_t row_stride, std::ptrdiff_t column_stride>
constexpr auto operator-(const MatrixView<num_rows, num_columns, const T, row_stride, column_stride> m) noexcept
{
    Matrix<num_rows, num_columns, T> res;
    std::transform(m.begin(), m.end(), res.begin(), std::negate<>{});
    return res;
}
// Negation operation for other Matrix types decay to MatrixView
constexpr auto operator-(const MatrixType auto& m) { return -m.cview(); }

// Helper macro to define arithmetic (+, -, *) scalar x Matrix, Matrix x scalar operations
#define DEFINE_MATRIX_SCALAR_OP(op)                                                                                  \
template <size_t num_rows, size_t num_columns, typename T, std::ptrdiff_t row_stride, std::ptrdiff_t column_stride>  \
constexpr auto operator op##=(                                                                                       \
    MatrixView<num_rows, num_columns, T, row_stride, column_stride> m, const T value) noexcept                       \
{                                                                                                                    \
    for (auto&& e : m) { e op##= value; }                                                                            \
    return m;                                                                                                        \
}                                                                                                                    \
template <MatrixType M>                                                                                              \
constexpr auto&& operator op##=(M&& m, const typename std::remove_reference_t<M>::T value) {                         \
    m.view() op##= value;                                                                                            \
    return std::forward<M>(m);                                                                                       \
}                                                                                                                    \
template <MatrixType M>                                                                                              \
constexpr auto operator op(const M& m, const typename M::T value) noexcept { return Matrix{ m } op##= value; }       \
template <MatrixType M>                                                                                              \
constexpr auto operator op(const typename M::T value, const M& m) noexcept { return Matrix{ m } op##= value; }

DEFINE_MATRIX_SCALAR_OP(+)
DEFINE_MATRIX_SCALAR_OP(-)
DEFINE_MATRIX_SCALAR_OP(*)

#undef DEFINE_MATRIX_SCALAR_OP

// Row vector times column vector gives the dot product of the two
template <RowVectorType V1, ColumnVectorType V2> requires (V1::k_size == V2::k_size)
constexpr auto operator*(const V1& lhs, const V2& rhs) noexcept
{
    using V = decltype(lhs[0] * rhs[0]);

    return std::inner_product(std::begin(lhs), std::end(lhs), std::begin(rhs), V{});
}

// Matrix multiplication is dot product of lhs rows with rhs columns
template <MatrixType M1, MatrixType M2>
    requires (M1::k_num_columns == M2::k_num_rows) && ((M1::k_num_rows > 1) || (M2::k_num_columns > 1))
constexpr auto operator*(const M1& lhs, const M2& rhs) noexcept
{
    using V = decltype(lhs.row(0) * rhs.column(0));

    Matrix<M1::k_num_rows, M2::k_num_columns, V> res;

    for (size_t i = 0; i < res.k_num_rows; ++i) {
        for (size_t j = 0; j < res.k_num_columns; ++j) {
            res(i, j) = lhs.row(i) * rhs.column(j);
        }
    }

    return res;
}

// 3D-Vector cross-product gives orthogonal vector with magnitude equal to area of parallelogram
constexpr auto cross(const VectorTypeN<3> auto& vec1, const VectorTypeN<3> auto& vec2) noexcept
{
    using V = decltype(vec1[0] * vec2[0]);

    return Vector<3, V>({
        (vec1[1] * vec2[2]) - (vec1[2] * vec2[1]),
        (vec1[2] * vec2[0]) - (vec1[0] * vec2[2]),
        (vec1[0] * vec2[1]) - (vec1[1] * vec2[0]),
    });
}

// 1x1 Matrix inversion is equal to multiplication reciprocal
constexpr auto inverse(const SquareMatrixTypeN<1> auto& m)
{
    Matrix<1, 1, double> inv_m;

    inv_m[0] = 1. / m[0];

    return inv_m;
}

// 2x2 Matrix inversion
constexpr auto inverse(const SquareMatrixTypeN<2> auto& m)
{
    constexpr Matrix<2, 2, double> rotate90deg({
        0, 1,
        -1, 0
    });

    Matrix<2, 2, double> inv_m;

    // We could also write it as:
    //     inv_m.row(0) = m(1, 1), m(0, 1);
    //     inv_m.row(1) = -m(0, 0), m(0, 0);
    inv_m.row(0) = transpose(rotate90deg * m.column(1));
    inv_m.row(1) = transpose(-rotate90deg * m.column(0));

    const double determinant = inv_m.row(0) * m.column(0);

    return (1. / determinant) * inv_m;
}

// 3x3 Matrix inversion
constexpr auto inverse(const SquareMatrixTypeN<3> auto& m)
{
    Matrix<3, 3, double> inv_m;

    inv_m.row(0) = transpose(cross(m.column(1), m.column(2)));
    inv_m.row(1) = transpose(cross(m.column(2), m.column(0)));
    inv_m.row(2) = transpose(cross(m.column(0), m.column(1)));

    const double determinant = inv_m.row(0) * m.column(0);

    return (1. / determinant) * inv_m;
}
}   // namespace matlib

template <size_t num_rows, size_t num_columns, typename T>
using Matrix = matlib::Matrix<num_rows, num_columns, T>;

template <size_t num_rows, typename T>
using Vector = matlib::Vector<num_rows, T>;

#endif /* INCLUDE_MATLIB_H_ */
